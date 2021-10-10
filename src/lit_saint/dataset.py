from typing import List, Tuple

from einops import rearrange
import pandas as pd
from sklearn.base import TransformerMixin
from torch.utils.data import Dataset
from torch import Tensor
import torch


class SaintDataset(Dataset):
    """Contains the data that will be processed in batches, and divide them in categorical and continuous

    :param data: Dataframe containing the data to use for the batches
    :param target: Name of the target column
    :param cat_cols: List of indices of categorical columns
    :param target_categorical: If True the target is categorical, so it is a classification problem
    :param con_cols: List of indices of continuous columns
    :param scaler: a scikit learn scaler used to transform the continuous columns
    """
    def __init__(self, data: pd.DataFrame, target: str, cat_cols: List[str],
                 con_cols: List[str], scaler: TransformerMixin, target_categorical: bool):
        self.X_categorical = self._define_tensor_features(data, cat_cols, torch.int64)
        self.X_continuos = self._define_tensor_features(data, con_cols, torch.float32, scaler)
        self.y = self._define_tensor_target(data, target, target_categorical)

    def __len__(self):
        return len(self.y)

    @staticmethod
    def _define_tensor_features(df: pd.DataFrame, cols: List[str], dtype: torch.dtype,
                                transformer: TransformerMixin = None) -> Tensor:
        """It convert a ndarray fo features in a Tensor

        :param df: Contains the data to put insert in the tensor
        :param cols: list of column names used to selected the data
        :param dtype: The type of returned Tensor
        """
        if len(cols) > 0:
            df = df.loc[:, cols].values
            if transformer:
                df = transformer.transform(df)
            return torch.from_numpy(df).to(dtype=dtype)
        else:
            return rearrange(torch.zeros(df.shape[0], dtype=dtype), 'n -> n 1')

    @staticmethod
    def _define_tensor_target(df: pd.DataFrame, target: str, target_categorical: bool) -> Tensor:
        """It return a Tensor containing the values of the target column

        :param df: Dataframe that contains the target
        :param target: name of the target column
        :param target_categorical: True if the target is categorical, otherwise is False
        """
        if target in df.columns:
            if target_categorical:
                y = torch.from_numpy(df[target].values).to(dtype=torch.int64)
            else:
                y = torch.from_numpy(df[target].values).to(dtype=torch.float32)
                y = rearrange(y, 'n -> n 1')
            return y
        else:
            return torch.zeros(df.shape[0], dtype=torch.float32)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor]:
        """It returns two tensors one for the categorical values and another for the continuous values.
        The target is concatenated as last column or of the categorical tensors or the continuous ones
        based on the type of the target

        :param idx: numeric index of the data that we want to process
        """
        return self.X_categorical[idx], self.X_continuos[idx], self.y[idx]
