import os
from pathlib import Path

import numpy as np
import wget
from hydra.core.config_store import ConfigStore

import pandas as pd

import hydra
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from hydra.utils import get_original_cwd
from pytorch_lightning import Trainer, seed_everything
import torchmetrics

from lit_saint import Saint, SaintConfig, SaintDatamodule, SaintTrainer

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=SaintConfig)


@hydra.main(config_path=".", config_name="config")
def read_config(cfg: SaintConfig) -> None:
    seed_everything(42, workers=True)
    df = pd.read_csv(get_original_cwd() + "/data/adult.csv", header=None)
    df_train, df_test = train_test_split(df, test_size=0.10, random_state=42)
    df_train, df_val = train_test_split(df_train, test_size=0.10, random_state=42)
    df_train["split"] = "train"
    df_val["split"] = "validation"
    df = pd.concat([df_train, df_val])
    data_module = SaintDatamodule(df=df, target=df.columns[14], split_column="split")
    model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                  config=cfg, dim_target=data_module.dim_target,
                  metrics_single_class=False)
    checkpoint_callback_pre = ModelCheckpoint(
        monitor="val_loss",
        dirpath=".",
        filename="saint-{epoch:02d}-{val_loss:.2f}",
        mode="min",
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3)

    pretrainer = Trainer(max_epochs=1, callbacks=[checkpoint_callback_pre, early_stopping_callback])
    trainer = Trainer(max_epochs=3, callbacks=[],
                      deterministic=True)
    saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer,
                                 pretrain_loader_params={"batch_size": cfg.pretrain.batch_size,
                                                         "num_workers": cfg.network.num_workers},
                                 train_loader_params={"batch_size": cfg.train.batch_size,
                                                         "num_workers": cfg.network.num_workers}
                                 )
    saint_trainer.fit(model=model, datamodule=data_module, enable_pretraining=False)
    output = saint_trainer.predict(model=model, datamodule=data_module, df=df_test,
                                   feature_importance=True)
    prediction = output["prediction"]
    feature_importance = output["importance"]
    df_test["prediction"] = np.argmax(prediction, axis=1)
    print(classification_report(data_module.predict_set[df.columns[14]], df_test["prediction"]))
    print(feature_importance)


if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    out = Path(os.getcwd() + '/data/adult.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("File already exists.")
    else:
        wget.download(url, out.as_posix())
    read_config()
