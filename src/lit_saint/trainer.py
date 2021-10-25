from typing import Dict

import numpy
import torch
from pandas import DataFrame

from lit_saint import Saint, SaintDatamodule
from pytorch_lightning.trainer import Trainer


class SaintTrainer:
    """This class simplify how to train and make predictions using Saint

    :param pretrainer: The Trainer that will be used during the pretraining phase
    :param trainer: The Trainer that will be used during the training and prediction phase
    :param train_loader_params: parameters used to configure the DataLoader during the training phase
    :param pretrain_loader_params: parameters used to configure the DataLoader during the pretraining phase
    """
    def __init__(self, pretrainer: Trainer = None, trainer: Trainer = None, train_loader_params: Dict = None,
                 pretrain_loader_params: Dict = None):
        self.pretrainer = pretrainer
        self.pretrain_loader_params = pretrain_loader_params if pretrain_loader_params else {"batch_size": 256}
        self.trainer = trainer
        self.train_loader_params = train_loader_params if train_loader_params else {"batch_size": 256}

    def prefit(self, model: Saint, datamodule: SaintDatamodule) -> None:
        """Function that is used for the pretraining of the model

        :param model: instance of Saint that will be used for the pretraining
        :param datamodule: instance of SaintDataModule that contains the data for the pretraining
        """
        model.set_pretraining(True)
        datamodule.set_pretraining(True)
        datamodule.set_data_loader_params(self.pretrain_loader_params)
        self.pretrainer.fit(model=model, datamodule=datamodule)

    def get_model_from_checkpoint(self, model: Saint, pretraining: bool) -> None:
        """Function that load the best checkpoint and update the model inplace

        :param model: instance of Saint that will be updated by the checkpoint
        :param pretraining: boolean flag if true the checkpoint will be taken from the pretrainer
        """
        trainer = self.pretrainer if pretraining else self.trainer
        checkpoint_callback = [c for c in trainer.callbacks if c.__class__.__name__ == 'ModelCheckpoint']
        if len(checkpoint_callback) == 1 and checkpoint_callback[0].best_model_path != "":
            model.load_from_checkpoint(checkpoint_path=checkpoint_callback[0].best_model_path)

    def fit(self, model: Saint, datamodule: SaintDatamodule, enable_pretraining: bool) -> None:
        """Runs the full optimization routine.

        :param model: instance of Saint that will be used for the training
        :param datamodule: instance of SaintDataModule that contains the data for the training
        :param enable_pretraining: boolean flag, if True the pretraining will be executed otherwise only the training
        """
        if enable_pretraining:
            self.prefit(model=model, datamodule=datamodule)
            self.get_model_from_checkpoint(model, pretraining=True)
        model.set_pretraining(False)
        datamodule.set_pretraining(False)
        datamodule.set_data_loader_params(self.train_loader_params)
        self.trainer.fit(model=model, datamodule=datamodule)
        self.get_model_from_checkpoint(model, pretraining=False)

    def predict(self, model: Saint, datamodule: SaintDatamodule, df: DataFrame,
                mc_dropout_iterations: int = 0) -> numpy.ndarray:
        """Separates from fit to make sure you never run on your predictions set until you want to.
        This will call the model forward function to compute predictions.

        :param model: model used for the prediction
        :param datamodule: the datamodule that will preprocess the data before prediction
        :param df: the data taht will be used for prediction
        :param mc_dropout_iterations: number of iterations used for mcdropout estimation, if 0 it will not use
        mcdropout
        """
        datamodule.set_predict_set(df)
        if mc_dropout_iterations > 0:
            model.set_mcdropout(True)
            mc_predictions = []
            for _ in range(mc_dropout_iterations):
                prediction = torch.cat(self.trainer.predict(model, datamodule=datamodule))
                mc_predictions.append(prediction)
            model.set_mcdropout(False)
            return torch.stack(mc_predictions, axis=2).cpu().numpy()
        prediction = self.trainer.predict(model, datamodule=datamodule)
        return torch.cat(prediction).cpu().numpy()
