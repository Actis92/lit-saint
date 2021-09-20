from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from lit_saint.dataset import SaintDataset


class SaintDatamodule(LightningDataModule):
    """It preprocess the data, doing LabelEncoding for the categorical values and fitting a StandardScaler
    for the numerical columns on the training set. And it splits the data and defines the dataloaders
    """
    def __init__(self, df: pd.DataFrame, target: str, split_column: str, batch_size: int = 256):
        """
        :param df: contains the data that will be used by the dataloaders
        :param target: name of the target column
        :param split_column: name of the column used to split the data
        :param batch_size: dimension of the batches
        """
        super().__init__()
        self.target: str = target
        self.batch_size = batch_size
        self.categorical_columns = []
        self.categorical_dims = []
        self.numerical_columns = []
        self.target_categorical = False
        self.scaler = StandardScaler()
        self.prep(df, split_column)
        self._split_data(df=df, split_column=split_column)
        self.scaler_continuous_columns(df=df, split_column=split_column)

    def prep(self, df: pd.DataFrame, split_column: str) -> None:
        """It find the indexes for each categorical and continuous columns, and for each categorical it
            applies Label Encoding in order to convert them in integers and save the number of classes for each
            categorical column

        :param df: contains the data that need to be processed
        :param split_column: name of column used to split the data
        """
        dim_target = None
        col_not_to_use = []
        for i, col in enumerate(df.columns):
            if df[col].dtypes.name in ["object", "category"]:
                if col != split_column:
                    l_enc = LabelEncoder()
                    df[col] = l_enc.fit_transform(df[col].values)
                    if col == self.target:
                        self.target_categorical = True
                        dim_target = len(l_enc.classes_)
                    else:
                        self.categorical_columns.append(i)
                        self.categorical_dims.append(len(l_enc.classes_))
            elif df[col].dtypes.name in ["int64", "float64", "int32", "float32"]:
                if col != self.target:
                    self.numerical_columns.append(i)
            else:
                col_not_to_use.append(col)
        if len(self.categorical_columns) == 0:
            self.categorical_dims.append(1)
        if self.target_categorical:
            self.categorical_dims.append(dim_target)
        print("The following cols will not be used because they have a not supported data type: ", col_not_to_use)

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
