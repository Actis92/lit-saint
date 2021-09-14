import numpy as np
from torch.utils.data import Dataset


class SaintDataset(Dataset):
    def __init__(self, data, target, is_pretraining, cat_cols, target_categorical, target_index, scaler):
        con_cols = list(set(np.arange(data.shape[1])) - set(cat_cols))
        if target_categorical:
            cat_cols.remove(target_index)
        else:
            con_cols.remove(target_index)
        self.is_pretraining = is_pretraining
        self.target_categorical = target_categorical
        self.X1 = data.iloc[:, cat_cols].values.astype(np.int64)  # categorical columns
        self.X2 = scaler.fit_transform(data.iloc[:, con_cols].values).astype(np.float32)  # numerical columns
        self.y = data[target].values.reshape(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        if self.is_pretraining:
            if self.target_categorical:
                return np.concatenate((self.X1[idx], self.y[idx])), self.X2[idx]
            else:
                return self.X1[idx], np.concatenate((self.X2[idx], self.y[idx].astype(np.float32)))
        else:
            return self.X1[idx], self.X2[idx], self.y[idx]