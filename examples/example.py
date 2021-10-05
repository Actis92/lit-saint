import os
from pathlib import Path

import torch
import wget
from hydra.core.config_store import ConfigStore

import pandas as pd

import hydra
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from hydra.utils import get_original_cwd
from pytorch_lightning import Trainer

from src.lit_saint import SAINT, SaintConfig, SaintDatamodule

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=SaintConfig)


@hydra.main(config_path=".", config_name="config")
def read_config(cfg: SaintConfig) -> None:
    df = pd.read_csv(get_original_cwd() + "/data/adult.csv")
    df_train, df_test = train_test_split(df, test_size=0.10, random_state=42)
    df_train, df_val = train_test_split(df_train, test_size=0.10, random_state=42)
    df_train["split"] = "train"
    df_val["split"] = "validation"
    df = pd.concat([df_train, df_val])
    data_module = SaintDatamodule(df=df, target=df.columns[14], split_column="split", pretraining=True)
    model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                  config=cfg, pretraining=True, dim_target=data_module.dim_target)
    pretrainer = Trainer(max_epochs=1, callbacks=[EarlyStopping(monitor="validation_loss", min_delta=0.00, patience=3)])
    pretrainer.fit(model, data_module)
    model.pretraining = False
    data_module.pretraining = False
    trainer = Trainer(max_epochs=1, callbacks=[EarlyStopping(monitor="validation_loss", min_delta=0.00, patience=3)])
    trainer.fit(model, data_module)
    data_module.set_predict_set(df_test)
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    prediction = trainer.predict(model, datamodule=data_module)
    prediction2 = trainer.predict(model, datamodule=data_module)
    df_test["prediction"] = torch.cat(prediction).numpy()
    print(classification_report(data_module.predict_set[df.columns[14]], df_test["prediction"]))


if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    out = Path(os.getcwd() + '/data/adult.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("File already exists.")
    else:
        wget.download(url, out.as_posix())
    read_config()
