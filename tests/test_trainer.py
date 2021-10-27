import numpy as np
import pandas as pd
import torchmetrics
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer

from lit_saint import SaintDatamodule, Saint, SaintConfig, SaintTrainer
from src.lit_saint.config import ContrastiveEnum, DenoisingEnum

cs = ConfigStore.instance()
cs.store(name="base_config", node=SaintConfig)


def test_train():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_cont": [2, 3, 1, 4],
                           "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split")
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
        data_module = SaintDatamodule(df=df, target="target", split_column="split")
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
        data_module = SaintDatamodule(df=df, target="target", split_column="split")
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
        data_module = SaintDatamodule(df=df, target="target", split_column="split")
        model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        saint_trainer = SaintTrainer(None, trainer=trainer)
        saint_trainer.fit(model, data_module, enable_pretraining=False)
        df_predict = df[[col for col in df.columns if col != "target"]]
        prediction = saint_trainer.predict(model, datamodule=data_module, df=df_predict)
        assert prediction.shape[1] == 2


def test_predict_unknown_categ():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_categ": ["a", "b", "a", "b"],
                           "feat_cont": [2, 3, 1, 4], "split": ["train", "train", "validation", "test"]})
        df_test = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_categ": ["a", "c", "d", "b"],
                                "feat_cont": [2, 3, 1, 4]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split")
        model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        saint_trainer = SaintTrainer(None, trainer=trainer)
        saint_trainer.fit(model, data_module, enable_pretraining=False)
        prediction = saint_trainer.predict(model, datamodule=data_module, df=df_test)
        assert prediction.shape[1] == 2


def test_regression():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": [1, 2, 3, 4], "feat_cont": [2, 3, 1, 4],
                           "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split")
        model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer)
        saint_trainer.fit(model, data_module, enable_pretraining=True)
        df_predict = df[[col for col in df.columns if col != "target"]]
        prediction = saint_trainer.predict(model, datamodule=data_module, df=df_predict)
        assert prediction.shape[1] == 1


def test_train_default_value_config():
    saint_cfg = SaintConfig()
    df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_cont": [2, 3, 1, 4],
                       "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split")
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
        data_module = SaintDatamodule(df=df, target="target", split_column="split")
        model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        saint_trainer = SaintTrainer(None, trainer=trainer)
        saint_trainer.fit(model, data_module, enable_pretraining=False)
        df_predict = df[[col for col in df.columns if col != "target"]]
        prediction = saint_trainer.predict(model, datamodule=data_module, df=df_predict)
        assert prediction.shape[1] == 3


def test_metrics_single_class():
    saint_cfg = SaintConfig()
    df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_cont": [2, 3, 1, 4],
                       "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split")
    model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                  config=saint_cfg, dim_target=data_module.dim_target,
                  metrics={"f1_score": torchmetrics.F1(num_classes=2, average=None)},
                  metrics_single_class=True)
    pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer)
    saint_trainer.fit(model, data_module, enable_pretraining=True)


def test_metrics_global():
    saint_cfg = SaintConfig()
    df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_cont": [2, 3, 1, 4],
                       "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split")
    model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                  config=saint_cfg, dim_target=data_module.dim_target, metrics={"f1_score": torchmetrics.F1()},
                  metrics_single_class=False)
    pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer)
    saint_trainer.fit(model, data_module, enable_pretraining=True)


def test_metrics_regression():
    saint_cfg = SaintConfig()
    df = pd.DataFrame({"target": [1, 2, 3, 4], "feat_cont": [2, 3, 1, 4],
                       "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split")
    model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                  config=saint_cfg, dim_target=data_module.dim_target,
                  metrics={"mae_score": torchmetrics.MeanAbsoluteError()},
                  metrics_single_class=False)
    pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer)
    saint_trainer.fit(model, data_module, enable_pretraining=True)


def test_mcdropout():
    saint_cfg = SaintConfig()
    df = pd.DataFrame({"target": ["0", "1", "1", "0"],
                       "feat_cont": [2, 3, 1, 4], "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split")
    model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                  config=saint_cfg, dim_target=data_module.dim_target)
    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    saint_trainer = SaintTrainer(None, trainer=trainer)
    saint_trainer.fit(model, data_module, enable_pretraining=False)
    df_predict = df[[col for col in df.columns if col != "target"]]
    prediction = saint_trainer.predict(model, datamodule=data_module, df=df_predict, mc_dropout_iterations=2)
    var_prediction = np.var(prediction, axis=2)
    assert prediction.shape[1] == 2
    assert prediction.shape[2] == 2
    assert var_prediction.min() > 0


def test_custom_params_dataloader():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        saint_cfg = SaintConfig(**cfg)
        df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_cont": [2, 3, 1, 4],
                           "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
        data_module = SaintDatamodule(df=df, target="target", split_column="split")
        model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                      config=saint_cfg, dim_target=data_module.dim_target)
        pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer,
                                     pretrain_loader_params={"batch_size": saint_cfg.pretrain.batch_size,
                                                             "num_workers": 0},
                                     train_loader_params={"batch_size": saint_cfg.train.batch_size,
                                                          "num_workers": 0})
        saint_trainer.prefit(model, data_module)
        assert data_module.data_loader_params['batch_size'] == 256
        assert data_module.data_loader_params['num_workers'] == 0
        saint_trainer.fit(model, data_module, enable_pretraining=False)
        assert data_module.data_loader_params['batch_size'] == 32
        assert data_module.data_loader_params['num_workers'] == 0


def test_contrastive_disabled():
    saint_cfg = SaintConfig()
    saint_cfg.pretrain.task.contrastive.contrastive_type = ContrastiveEnum.disabled
    df = pd.DataFrame({"target": [1, 2, 3, 4], "feat_cont": [2, 3, 1, 4],
                       "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split")
    model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                  config=saint_cfg, dim_target=data_module.dim_target,
                  metrics={"mae_score": torchmetrics.MeanAbsoluteError()},
                  metrics_single_class=False)
    pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer)
    saint_trainer.fit(model, data_module, enable_pretraining=True)


def test_denoising_disabled():
    saint_cfg = SaintConfig()
    saint_cfg.pretrain.task.denoising.denoising_type = DenoisingEnum.disabled
    df = pd.DataFrame({"target": [1, 2, 3, 4], "feat_cont": [2, 3, 1, 4],
                       "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split")
    model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                  config=saint_cfg, dim_target=data_module.dim_target,
                  metrics={"mae_score": torchmetrics.MeanAbsoluteError()},
                  metrics_single_class=False)
    pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer)
    saint_trainer.fit(model, data_module, enable_pretraining=True)