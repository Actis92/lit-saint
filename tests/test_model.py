import pandas as pd
import torch
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer

from lit_saint import SaintDatamodule, SAINT, SaintConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=SaintConfig)


def test_train():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_cont": [2, 3, 1, 4],
                           "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split")
        model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, pretraining=True)
        pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
        pretrainer.fit(model, data_module)
        model.pretraining = False
        data_module.pretraining = False
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer.fit(model, data_module)


def test_train_no_continuos_columns():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0"],
                           "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split")
        model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, pretraining=True)
        pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
        pretrainer.fit(model, data_module)
        model.pretraining = False
        data_module.pretraining = False
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer.fit(model, data_module)


def test_train_no_categorical_columns():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0"],
                           "feat_cont": [2, 3, 1, 4], "split": ["train", "train", "validation", "test"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split")
        model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, pretraining=True)
        pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
        pretrainer.fit(model, data_module)
        model.pretraining = False
        data_module.pretraining = False
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer.fit(model, data_module)


def test_predict():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0"],
                           "feat_cont": [2, 3, 1, 4], "split": ["train", "train", "validation", "test"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split")
        model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer.fit(model, data_module)
        data_module.set_predict_set(df[[col for col in df.columns if col != "target"]])
        prediction = trainer.predict(model, datamodule=data_module)
        df["prediction"] = torch.cat(prediction).numpy()


def test_predict_unknown_categ():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_categ": ["a", "b", "a", "b"],
                           "feat_cont": [2, 3, 1, 4], "split": ["train", "train", "validation", "test"]})
        df_test = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_categ": ["a", "c", "d", "b"],
                                "feat_cont": [2, 3, 1, 4]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split")
        model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer.fit(model, data_module)
        data_module.set_predict_set(df_test)
        prediction = trainer.predict(model, datamodule=data_module)
        df_test["prediction"] = torch.cat(prediction).numpy()