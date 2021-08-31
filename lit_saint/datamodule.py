from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from lit_saint.dataset import SaintDataset


class SaintDatamodule(LightningDataModule):
    def __init__(self, df: pd.DataFrame, target: str, split_column: str, batch_size: int = 10,
                 pretraining: bool = False):
        super().__init__()
        self.target = target
        self.batch_size = batch_size
        self.categorical_columns = []
        self.categorical_dims = []
        self.num_continuos = 0
        self.target_categorical = False
        self.prep(df, split_column)
        self._split_data(df=df, split_column=split_column)
        self.pretraining = pretraining

    def prep(self, df, split_column):
        for i, col in enumerate(df.columns):
            if df[col].dtypes == object:
                if col != split_column:
                    l_enc = LabelEncoder()
                    df[col] = l_enc.fit_transform(df[col].values)
                    self.categorical_columns.append(i)
                    self.categorical_dims.append(len(l_enc.classes_))
                if col == self.target:
                    self.target_categorical = True
        self.num_continuos = df.shape[1] - len(self.categorical_columns)

    def _split_data(self, df, split_column):
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
            is_pretraining=self.pretraining,
            cat_cols=self.categorical_columns,
            target_categorical=self.target_categorical
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
            is_pretraining=self.pretraining,
            cat_cols=self.categorical_columns,
            target_categorical=self.target_categorical
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
                is_pretraining=self.pretraining,
                cat_cols=self.categorical_columns,
                target_categorical=self.target_categorical
            )
            return DataLoader(
                dataset,
                self.batch_size
            )
