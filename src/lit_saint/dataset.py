from typing import List, Tuple

from einops import rearrange
import numpy as np
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
    def __init__(self, data: pd.DataFrame, target: str, cat_cols: List[int],
                 target_categorical: bool, con_cols: List[int], scaler: TransformerMixin):
        self.target_categorical = target_categorical
        if len(cat_cols) > 0:
            self.X_categorical: Tensor = torch.from_numpy(data.iloc[:, cat_cols].values.astype(np.int64))
        else:
            self.X_categorical: Tensor = rearrange(torch.zeros(data.shape[0], dtype=torch.int64), 'n -> n 1')
        if len(con_cols) > 0:
            self.X_continuos = torch.from_numpy(scaler.fit_transform(data.iloc[:, con_cols].values).astype(np.float32))
        else:
            self.X_continuos: Tensor = rearrange(torch.zeros(data.shape[0], dtype=torch.float32), 'n -> n 1')
        self.y = torch.from_numpy(data[target].values.reshape(-1, 1))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        """It returns two tensors one for the categorical values and another for the continuous values.
        The target is concatenated as last column or of the categorical tensors or the continuous ones
        based on the type of the target

        :param idx: numeric index of the data that we want to process
        """
        if self.target_categorical:
            return torch.cat((self.X_categorical[idx], self.y[idx])), self.X_continuos[idx]
        else:
            return self.X_categorical[idx], torch.cat((self.X_continuos[idx], self.y[idx].astype(np.float32)))
