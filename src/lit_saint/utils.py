import pandas as pd
import torch
from pytorch_lightning import Trainer
from torch import Tensor

from lit_saint import SaintDatamodule, SAINT


def pretraining_and_training_model(data_module: SaintDatamodule, model: SAINT, pretrainer: Trainer = None,
                                   trainer: Trainer = None) -> [SAINT, Trainer]:
    """Tis utility allow to execute the pretraining step and the training one or only one of them

    :param data_module:
    :param model:
    :param pretrainer:
    :param trainer:
    :return:
    """
    if pretrainer:
        pretrainer.fit(model, data_module)
    model.pretraining = False
    data_module.pretraining = False
    if trainer:
        trainer.fit(model, data_module)
        return model, trainer
    return model, pretrainer


def mc_dropout(data_module: SaintDatamodule, model: SAINT, trainer: Trainer, n_iterations: int,
               df: pd.DataFrame) -> Tensor:
    """

    :param data_module:
    :param model:
    :param trainer:
    :param n_iterations:
    :param df:
    :return:
    """
    data_module.set_predict_set(df)
    model.mc_dropout = True
    mc_predictions = []
    for i in range(n_iterations):
        prediction = torch.cat(trainer.predict(model, datamodule=data_module))
        mc_predictions.append(prediction)
    model.mc_dropout = False
    return torch.stack(mc_predictions, axis=2)
