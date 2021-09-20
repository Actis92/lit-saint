from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from lit_saint.dataset import SaintDataset


class SaintDatamodule(LightningDataModule):
    """It preprocess the data, doing LabelEncoding for the categorical values and fitting a StandardScaler
    for the numerical columns on the training set. And it splits the data and defines the dataloaders
    """
    def __init__(self, df: pd.DataFrame, target: str, split_column: str, batch_size: int = 256,
                 scaler: TransformerMixin = None, pretraining: bool = False):
        """
        :param df: contains the data that will be used by the dataloaders
        :param target: name of the target column
        :param split_column: name of the column used to split the data
        :param batch_size: dimension of the batches
        :param pretraining: boolean flag, if False it use only were the target is not NaN
        """
        super().__init__()
        self.target: str = target
        self.pretraining = pretraining
        self.batch_size = batch_size
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
                    df[col] = df[col].fillna("SAINT_NAN")
                if col != split_column:
                    l_enc = LabelEncoder()
                    df[col] = l_enc.fit_transform(df[col].values)
                    self.dict_label_encoder[col] = l_enc
                    if col == self.target:
                        self.target_categorical = True
                        dim_target = len(l_enc.classes_)
                        self.target_nan_index = list(l_enc.classes_).index("SAINT_NAN") if 'SAINT_NAN' \
                                                                                       in l_enc.classes_ else None
                    else:
                        self.categorical_columns.append(i)
                        self.categorical_dims.append(len(l_enc.classes_))
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

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        df = self.train
        if not self.pretraining:
            df = df.loc[df[self.target] != self.target_nan_index]
            df[self.target] = df[self.target].apply(lambda x: x if x < self.target_nan_index else x - 1)
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
            self.batch_size
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        df = self.validation
        if not self.pretraining:
            df = df.loc[df[self.target] != self.target_nan_index]
            df[self.target] = df[self.target].apply(lambda x : x if x < self.target_nan_index else x - 1)
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
            self.batch_size
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        if self.test is not None:
            df = self.test
            if not self.pretraining:
                df = df.loc[df[self.target] != self.target_nan_index]
                df[self.target] = df[self.target].apply(lambda x: x if x < self.target_nan_index else x - 1)
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
                self.batch_size
            )

    def set_predict_set(self, df):
        for col, label_enc in self.dict_label_encoder.items():
            if df[col].isna().any():  # the columns contains nan
                df[col] = df[col].fillna("SAINT_NAN")
            df[col] = label_enc.transform(df[col].values)
        self.predict_set = df

    def predict_dataloader(self) -> DataLoader:
        df = self.predict_set
        if not self.pretraining:
            df = df.loc[df[self.target] != self.target_nan_index]
            df[self.target] = df[self.target].apply(lambda x: x if x < self.target_nan_index else x - 1)
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
            self.batch_size
        )
