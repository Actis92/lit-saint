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

    def __init__(self, df: pd.DataFrame, target: str, split_column: str, scaler: TransformerMixin = None):
        """
        :param df: contains the data that will be used by the dataLoaders
        :param target: name of the target column
        :param split_column: name of the column used to split the data
        :param scaler: a scikit learn transformer in order to rescale the continuos variables, if not specified
        it will use the StandardScaler
        """
        super().__init__()
        self.target: str = target
        self.pretraining = False
        self.data_loader_params = {"batch_size": 256}
        self.categorical_columns = []
        self.categorical_dims = []
        self.numerical_columns = []
        self.dim_target = 1
        self.target_nan_index = None
        self.dict_label_encoder = {}
        self.predict_set = None
        self.scaler = scaler if scaler else StandardScaler()
        self.split_column = split_column
        self.prep(df)

    def prep(self, df: pd.DataFrame) -> None:
        """It find the indexes for each categorical and continuous columns, and for each categorical it
            applies Label Encoding in order to convert them in integers and save the number of classes for each
            categorical column

        :param df: contains the data that need to be processed
        """
        df = df.copy()
        col_not_to_use = []
        for col in df.columns:
            if df[col].dtypes.name in ["object", "category"]:
                df = self.prep_categorical_columns(col=col, df=df)
            elif df[col].dtypes.name in ["int64", "float64", "int32", "float32"]:
                df = self.prep_continuous_columns(col=col, df=df)
            else:
                col_not_to_use.append(col)
        if len(self.categorical_columns) == 0:
            self.categorical_dims.append(1)
        print("The following cols will not be used because they have a not supported data type: ", col_not_to_use)
        self._split_data(df=df)
        self.scaler_continuous_columns(df=df)

    def prep_categorical_columns(self, col: str, df: pd.DataFrame) -> pd.DataFrame:
        if df[col].isna().any():  # the columns contains nan
            df[col] = df[col].fillna(self.NAN_LABEL)
        if col != self.split_column:
            l_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=df[col].nunique())
            df[col] = l_enc.fit_transform(df[col].values.reshape(-1, 1)).astype(int)
            self.dict_label_encoder[col] = l_enc
            if col == self.target:
                if self.NAN_LABEL in l_enc.categories_[0]:
                    self.dim_target = len(l_enc.categories_[0]) - 1
                    self.target_nan_index = list(l_enc.categories_[0]).index(self.NAN_LABEL)
                else:
                    self.dim_target = len(l_enc.categories_[0])
                    self.target_nan_index = None
            else:
                self.categorical_columns.append(col)
                self.categorical_dims.append(len(l_enc.categories_[0]) + 1)
        return df

    def prep_continuous_columns(self, col: str, df: pd.DataFrame) -> pd.DataFrame:
        if df[col].isna().any():  # the columns contains nan
            df[col] = df[col].fillna(0)
        if col != self.target:
            self.numerical_columns.append(col)
        return df

    def scaler_continuous_columns(self, df: pd.DataFrame) -> None:
        """Fit a StandardScaler for each continuos columns on the training set

        :param df: contains the data that need to be processed
        :param split_column: name of column used to split the data
        """
        df_train = df.loc[df[self.split_column] == "train"].loc[:, self.numerical_columns].values
        if len(self.numerical_columns) > 0:
            self.scaler.fit(df_train)

    def _split_data(self, df: pd.DataFrame) -> None:
        """Split the Dataframe in train, validation and test, and drop the split column

        :param df: contains the data that need to be processed
        """
        self.train = df.loc[df[self.split_column] == "train"].reset_index(drop=True)
        self.validation = df.loc[df[self.split_column] == "validation"].reset_index(drop=True)
        self.test = df.loc[df[self.split_column] == "test"].reset_index(drop=True)
        self.train.drop(self.split_column, axis=1, inplace=True)
        self.validation.drop(self.split_column, axis=1, inplace=True)
        self.test.drop(self.split_column, axis=1, inplace=True)

    def set_predict_set(self, df) -> None:
        """Tranform the categorical columns using the OrdinalEncoders fitted before the training and
        save the dataframe in order to make the predictions

        :param df: The data that will be used to make some predictions
        """
        df = df.copy()
        for col, label_enc in self.dict_label_encoder.items():
            if col != self.target or (col == self.target and col in df.columns):
                if df[col].isna().any():  # the columns contains nan
                    df[col] = df[col].fillna(self.NAN_LABEL)
                df[col] = label_enc.transform(df[col].values.reshape(-1, 1)).astype(int)
        df = df.fillna(0)
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
            con_cols=self.numerical_columns,
            scaler=self.scaler,
            target_categorical=self.dim_target > 1
        )
        return DataLoader(
            dataset,
            **self.data_loader_params,
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

    def set_pretraining(self, pretraining: bool) -> None:
        """Function used to set the pretraining flag"""
        self.pretraining = pretraining

    def set_data_loader_params(self, data_loader_params: Dict) -> None:
        """Function used to set the parameters used by the DataLoader"""
        self.data_loader_params = data_loader_params
