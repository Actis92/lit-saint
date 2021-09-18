from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.lit_saint.dataset import SaintDataset


class SaintDatamodule(LightningDataModule):
    def __init__(self, df: pd.DataFrame, target: str, split_column: str, batch_size: int = 256):
        super().__init__()
        self.target = target
        self.batch_size = batch_size
        self.categorical_columns = []
        self.categorical_dims = []
        self.numerical_columns = []
        self.target_categorical = False
        self.scaler = StandardScaler()
        self.prep(df, split_column)
        self._split_data(df=df, split_column=split_column)
        self.scaler_continuos_columns(df=df, split_column=split_column)

    def prep(self, df: pd.DataFrame, split_column: str):
        """It find the indexes for each categorical and continuos columns, and for each categorical it
            applies Label Encoding in order to convert them in integers and save the number of classes for each
            categorical column"""
        for i, col in enumerate(df.columns):
            if df[col].dtypes.name in ["object", "category"]:
                if col != split_column:
                    l_enc = LabelEncoder()
                    df[col] = l_enc.fit_transform(df[col].values)
                    if col == self.target:
                        self.target_categorical = True
                    else:
                        self.categorical_columns.append(i)
                    self.categorical_dims.append(len(l_enc.classes_))
            else:
                if col != self.target:
                    self.numerical_columns.append(i)

    def scaler_continuos_columns(self, df: pd.DataFrame, split_column: str):
        """Fit a StandardScaler for each continuos columns on the training set"""
        df_train = df.loc[df[split_column] == "train"].iloc[:, self.numerical_columns].values
        self.scaler.fit(df_train)

    def _split_data(self, df: pd.DataFrame, split_column: str):
        """Split the Dataframe in train, validation and test, and drop the split column"""
        self.train = df.loc[df[split_column] == "train"]
        self.validation = df.loc[df[split_column] == "validation"]
        self.test = df.loc[df[split_column] == "test"]
        self.train.drop(split_column, axis=1, inplace=True)
        self.validation.drop(split_column, axis=1, inplace=True)
        self.test.drop(split_column, axis=1, inplace=True)

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        dataset = SaintDataset(
            data=self.train,
            target=self.target,
            cat_cols=self.categorical_columns,
            target_categorical=self.target_categorical,
            con_cols=self.numerical_columns,
            scaler = self.scaler
        )
        return DataLoader(
            dataset,
            self.batch_size
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        dataset = SaintDataset(
            data=self.validation,
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
            dataset = SaintDataset(
                data=self.test,
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
