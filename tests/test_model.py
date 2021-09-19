import pandas as pd
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
        model = SAINT(categories=data_module.categorical_dims, num_continuous=len(data_module.numerical_columns),
                      config=saint_cfg, pretraining=True)
        pretrainer = Trainer(max_epochs=1, fast_dev_run=True)
        pretrainer.fit(model, data_module)
        model.pretraining = False
        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer.fit(model, data_module)