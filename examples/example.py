import os
from pathlib import Path

import wget
from hydra.core.config_store import ConfigStore

import pandas as pd

import hydra
from hydra.utils import get_original_cwd
from pytorch_lightning import Trainer

from src.lit_saint import SAINT, SaintConfig, SaintDatamodule

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="base_config", node=SaintConfig)


@hydra.main(config_path=".", config_name="config")
def read_config(cfg: SaintConfig) -> None:
    df = pd.read_csv(get_original_cwd() + "/data/adult.csv")
    df["split"] = "train"
    df["split"].iloc[2000:3000] = "validation"
    df["split"].iloc[3000:] = "test"
    data_module = SaintDatamodule(df=df, target=df.columns[14], split_column="split")
    model = SAINT(categories=data_module.categorical_dims, num_continuous=len(data_module.numerical_columns),
                  config=cfg, pretraining=True)

    pretrainer = Trainer(max_epochs=3)
    pretrainer.fit(model, data_module)
    model.pretraining = False
    trainer = Trainer(max_epochs=3)
    trainer.fit(model, data_module)
    trainer.predict(model, data_module.test_dataloader())


if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    out = Path(os.getcwd() + '/data/adult.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("File already exists.")
    else:
        wget.download(url, out.as_posix())
    read_config()
