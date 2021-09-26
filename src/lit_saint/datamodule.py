import os
from typing import Dict

from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from lit_saint.dataset import SaintDataset


class SaintDatamodule(LightningDataModule):
    """It preprocess the data, doing LabelEncoding for the categorical values and fitting a StandardScaler
    for the numerical columns on the training set. And it splits the data and defines the dataloaders
    """
    NAN_LABEL = "SAINT_NAN"

    def __init__(self, df: pd.DataFrame, target: str, split_column: str, data_loader_params: Dict = None,
                 scaler: TransformerMixin = None, pretraining: bool = False):
        """
        :param df: contains the data that will be used by the dataLoaders
        :param target: name of the target column
        :param split_column: name of the column used to split the data
        :param data_loader_params: parameters used to configure the DataLoader
        :param pretraining: boolean flag, if False it use only where the target is not NaN
        """
        super().__init__()
        self.target: str = target
        self.pretraining = pretraining
        self.data_loader_params = data_loader_params if data_loader_params else {"batch_size": 256}
        self.categorical_columns = []
        self.categorical_dims = []
        self.numerical_columns = []
        self.target_categorical = False
        self.target_nan_index = None
        self.dict_label_encoder = {}
        self.predict_set = None
        self.scaler = scaler if scaler else StandardScaler()
        self.prep(df, split_column)

    def prep(self, df: pd.DataFrame, split_column: str) -> None:
        """It find the indexes for each categorical and continuous columns, and for each categorical it
            applies Label Encoding in order to convert them in integers and save the number of classes for each
            categorical column

        :param df: contains the data that need to be processed
        :param split_column: name of column used to split the data
        """
        df = df.copy()
        dim_target = None
        col_not_to_use = []
        for i, col in enumerate(df.columns):
            if df[col].dtypes.name in ["object", "category"]:
                if df[col].isna().any():  # the columns contains nan
                    df[col] = df[col].fillna(self.NAN_LABEL)
                if col != split_column:
                    l_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    df[col] = l_enc.fit_transform(df[col].values.reshape(-1, 1)).astype(int)
                    self.dict_label_encoder[col] = l_enc
                    if col == self.target:
                        self.target_categorical = True
                        dim_target = len(l_enc.categories_[0])

                        self.target_nan_index = list(l_enc.categories_[0]).index(self.NAN_LABEL) \
                            if self.NAN_LABEL in l_enc.categories_[0] else None
                    else:
                        self.categorical_columns.append(i)
                        self.categorical_dims.append(len(l_enc.categories_[0]) + 1)
            elif df[col].dtypes.name in ["int64", "float64", "int32", "float32"]:
                if df[col].isna().any():  # the columns contains nan
                    df[col] = df[col].fillna(0)
                if col != self.target:
                    self.numerical_columns.append(i)
            else:
                col_not_to_use.append(col)
        if len(self.categorical_columns) == 0:
            self.categorical_dims.append(1)
        if self.target_categorical:
            self.categorical_dims.append(dim_target)
        print("The following cols will not be used because they have a not supported data type: ", col_not_to_use)
        self._split_data(df=df, split_column=split_column)
        self.scaler_continuous_columns(df=df, split_column=split_column)

    def scaler_continuous_columns(self, df: pd.DataFrame, split_column: str) -> None:
        """Fit a StandardScaler for each continuos columns on the training set

        :param df: contains the data that need to be processed
        :param split_column: name of column used to split the data
        """
        df_train = df.loc[df[split_column] == "train"].iloc[:, self.numerical_columns].values
        if len(self.numerical_columns) > 0:
            self.scaler.fit(df_train)

    def _split_data(self, df: pd.DataFrame, split_column: str) -> None:
        """Split the Dataframe in train, validation and test, and drop the split column

        :param df: contains the data that need to be processed
        :param split_column: name of column used to split the data
        """
        self.train = df.loc[df[split_column] == "train"].reset_index(drop=True)
        self.validation = df.loc[df[split_column] == "validation"].reset_index(drop=True)
        self.test = df.loc[df[split_column] == "test"].reset_index(drop=True)
        self.train.drop(split_column, axis=1, inplace=True)
        self.validation.drop(split_column, axis=1, inplace=True)
        self.test.drop(split_column, axis=1, inplace=True)

    def set_predict_set(self, df) -> None:
        """Tranform the categorical columns using the OrdinalEncoders fitted before the training and
        save the dataframe in order to make the predictions

        :param df: The data that will be used to make some predictions
        """
        df = df.copy()
        for col, label_enc in self.dict_label_encoder.items():
            if df[col].isna().any():  # the columns contains nan
                df[col] = df[col].fillna(self.NAN_LABEL)
            df[col] = label_enc.fit_transform(df[col].values.reshape(-1, 1)).astype(int)
        self.predict_set = df

    def _remove_rows_without_labels(self, df) -> pd.DataFrame:
        """Remove rows from a dataframe where the label was NaN

        :param df: the dataframe from which remove rows
        """
        df = df.loc[df[self.target] != self.target_nan_index]
        df[self.target] = df[self.target].apply(lambda x: x if x < self.target_nan_index else x - 1)
        return df

    def _create_dataloader(self, df) -> DataLoader:
        """ Given a dataframe it return a dataloader and eventually without rows
        that have nan labels if not pretraining

        :param df: the dataframe that will be used inside the DataLoader
        """
        if not self.pretraining and self.target_nan_index is not None:
            df = self._remove_rows_without_labels(df)
        dataset = SaintDataset(
            data=df,
            target=self.target,
            cat_cols=self.categorical_columns,
            target_categorical=self.target_categorical,
            con_cols=self.numerical_columns,
            scaler=self.scaler
        )
        return DataLoader(
            dataset,
            num_workers=os.cpu_count(),
            **self.data_loader_params
        )

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return self._create_dataloader(self.train)

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return self._create_dataloader(self.validation)

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return self._create_dataloader(self.test)

    def predict_dataloader(self) -> DataLoader:
        """ Function that loads the dataset for the prediction. """
        return self._create_dataloader(self.predict_set)
