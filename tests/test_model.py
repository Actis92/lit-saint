import pandas as pd
import torch
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer

from lit_saint import SaintDatamodule, Saint, SaintConfig, SaintTrainer

cs = ConfigStore.instance()
cs.store(name="base_config", node=SaintConfig)


def test_train():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_cont": [2, 3, 1, 4],
                           "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
        model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer)
        saint_trainer.fit(model, data_module, enable_pretraining=True)


def test_train_no_continuous_columns():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0"],
                           "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
        model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer)
        saint_trainer.fit(model, data_module, enable_pretraining=True)


def test_train_no_categorical_columns():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0"],
                           "feat_cont": [2, 3, 1, 4], "split": ["train", "train", "validation", "test"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
        model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer)
        saint_trainer.fit(model, data_module, enable_pretraining=True)


def test_predict():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0"],
                           "feat_cont": [2, 3, 1, 4], "split": ["train", "train", "validation", "test"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
        model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        saint_trainer = SaintTrainer(None, trainer=trainer)
        saint_trainer.fit(model, data_module, enable_pretraining=False)
        df_predict = df[[col for col in df.columns if col != "target"]]
        prediction = saint_trainer.predict(model, datamodule=data_module, df=df_predict)
        prediction = prediction.numpy()
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
        model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        saint_trainer = SaintTrainer(None, trainer=trainer)
        saint_trainer.fit(model, data_module, enable_pretraining=False)
        prediction = saint_trainer.predict(model, datamodule=data_module, df=df_test).numpy()
        assert prediction.shape[1] == 2


def test_regression():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": [1, 2, 3, 4], "feat_cont": [2, 3, 1, 4],
                           "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
        model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer)
        saint_trainer.fit(model, data_module, enable_pretraining=True)
        df_predict = df[[col for col in df.columns if col != "target"]]
        prediction = saint_trainer.predict(model, datamodule=data_module, df=df_predict).numpy()
        assert prediction.shape[1] == 1


def test_train_default_value_config():
    saint_cfg = SaintConfig()
    df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_cont": [2, 3, 1, 4],
                       "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
    model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                  config=saint_cfg, dim_target=data_module.dim_target)
    pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer)
    saint_trainer.fit(model, data_module, enable_pretraining=True)


def test_multiclass():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0", "2", "2"],
                           "feat_cont": [2, 3, 1, 4, 5, 6], "split": ["train", "train", "validation", "val",
                                                                      "train", "validation"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split", num_workers=0)
        model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        saint_trainer = SaintTrainer(None, trainer=trainer)
        saint_trainer.fit(model, data_module, enable_pretraining=False)
        df_predict = df[[col for col in df.columns if col != "target"]]
        prediction = saint_trainer.predict(model, datamodule=data_module, df=df_predict).numpy()
        assert prediction.shape[1] == 3
