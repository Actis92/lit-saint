import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from lit_saint.dataset import SaintDataset


def test_get_item():
    scaler = StandardScaler()
    df = pd.DataFrame({"target": [0, 1], "feat_cont": [2, 1], "feat_categ": [1, 0]})
    scaler.fit(df.iloc[:, [1]].values)
    dataset = SaintDataset(data=df, target="target", cat_cols=[2], con_cols=[1], scaler=scaler, target_categorical=True)
    actual_batch = dataset.__getitem__(0)
    expected_batch = (torch.from_numpy(np.array([1, 0]).astype(np.int64)),
                      torch.from_numpy(np.array([1]).astype(np.float32)))
    assert torch.equal(expected_batch[0], actual_batch[0])
    assert torch.equal(expected_batch[1], actual_batch[1])
    assert dataset.target_categorical
    actual_batch = dataset.__getitem__(1)
    expected_batch = (torch.from_numpy(np.array([0, 1]).astype(np.int64)),
                      torch.from_numpy(np.array([-1]).astype(np.float32)))
    assert torch.equal(expected_batch[0], actual_batch[0])
    assert torch.equal(expected_batch[1], actual_batch[1])


def test_empty_categorical():
    scaler = StandardScaler()
    df = pd.DataFrame({"target": [0], "feat_cont": [2]})
    scaler.fit(df.iloc[:, [1]].values)
    dataset = SaintDataset(data=df, target="target", cat_cols=[], con_cols=[1], scaler=scaler, target_categorical=True)
    actual_batch = dataset.__getitem__(0)
    expected_batch = (torch.zeros(2, dtype=torch.int64), torch.zeros(1, dtype=torch.float32))
    assert torch.equal(expected_batch[0], actual_batch[0])
    assert torch.equal(expected_batch[1], actual_batch[1])
    assert dataset.target_categorical


def test_empty_continuous():
    scaler = StandardScaler()
    df = pd.DataFrame({"target": [0], "feat_cat": [0]})
    scaler.fit(df.iloc[:, [1]].values)
    dataset = SaintDataset(data=df, target="target", cat_cols=[1], con_cols=[], scaler=scaler, target_categorical=True)
    actual_batch = dataset.__getitem__(0)
    expected_batch = (torch.zeros(2, dtype=torch.int64), torch.zeros(1, dtype=torch.float32))
    assert torch.equal(expected_batch[0], actual_batch[0])
    assert torch.equal(expected_batch[1], actual_batch[1])
    assert dataset.target_categorical
