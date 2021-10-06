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
        data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
        model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, pretraining=True, dim_target=data_module.dim_target)
        pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
        pretrainer.fit(model, data_module)
        model.pretraining = False
        data_module.pretraining = False
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer.fit(model, data_module)


def test_train_no_continuous_columns():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0"],
                           "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
        model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, pretraining=True, dim_target=data_module.dim_target)
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
        data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
        model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, pretraining=True, dim_target=data_module.dim_target)
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
        data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
        model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer.fit(model, data_module)
        df_predict = df[[col for col in df.columns if col != "target"]]
        data_module.set_predict_set(df_predict)
        prediction = trainer.predict(model, datamodule=data_module)
        prediction = torch.cat(prediction).numpy()
        assert prediction.shape[1] == 2


def test_predict_unknown_categ():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_categ": ["a", "b", "a", "b"],
                           "feat_cont": [2, 3, 1, 4], "split": ["train", "train", "validation", "test"]})
        df_test = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_categ": ["a", "c", "d", "b"],
                                "feat_cont": [2, 3, 1, 4]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
        model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer.fit(model, data_module)
        data_module.set_predict_set(df_test)
        prediction = trainer.predict(model, datamodule=data_module)
        prediction = torch.cat(prediction).numpy()
        assert prediction.shape[1] == 2


def test_regression():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": [1, 2, 3, 4], "feat_cont": [2, 3, 1, 4],
                           "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
        model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, pretraining=True, dim_target=data_module.dim_target)
        pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
        pretrainer.fit(model, data_module)
        model.pretraining = False
        data_module.pretraining = False
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer.fit(model, data_module)
        df_predict = df[[col for col in df.columns if col != "target"]]
        data_module.set_predict_set(df_predict)
        prediction = trainer.predict(model, datamodule=data_module)
        prediction = torch.cat(prediction).numpy()
        assert prediction.shape[1] == 1


def test_train_default_value_config():
    saint_cfg = SaintConfig()
    df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_cont": [2, 3, 1, 4],
                       "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
    model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                  config=saint_cfg, pretraining=True, dim_target=data_module.dim_target)
    pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
    pretrainer.fit(model, data_module)
    model.pretraining = False
    data_module.pretraining = False
    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    trainer.fit(model, data_module)


def test_multiclass():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0", "2", "2"],
                           "feat_cont": [2, 3, 1, 4, 5, 6], "split": ["train", "train", "validation", "val",
                                                                      "train", "validation"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
        model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer.fit(model, data_module)
        df_predict = df[[col for col in df.columns if col != "target"]]
        data_module.set_predict_set(df_predict)
        prediction = trainer.predict(model, datamodule=data_module)
        prediction = torch.cat(prediction).numpy()
        assert prediction.shape[1] == 3
