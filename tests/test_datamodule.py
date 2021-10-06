from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler

from src.lit_saint import SaintDatamodule


def test_datamodule_target_categorical():
    df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_cont": [2, 3, 1, 4],
                       "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split")
    expected_train = pd.DataFrame({"target": [0, 1], "feat_cont": [2, 3], "feat_categ": [0, 1]})
    expected_validation = pd.DataFrame({"target": [1], "feat_cont": [1], "feat_categ": [0]})
    expected_test = pd.DataFrame({"target": [0], "feat_cont": [4], "feat_categ": [2]})
    pd.testing.assert_frame_equal(data_module.train, expected_train)
    pd.testing.assert_frame_equal(data_module.validation, expected_validation)
    pd.testing.assert_frame_equal(data_module.test, expected_test)
    assert data_module.categorical_columns == ["feat_categ"]
    assert data_module.numerical_columns == ["feat_cont"]
    # the target is always the last column and added one category in order to handling unknown
    assert data_module.categorical_dims == [4]
    check_is_fitted(data_module.scaler)


def test_datamodule_target_continuous():
    df = pd.DataFrame({"target": [1., 2., 3., 4.], "feat_cont": [2, 3, 1, 4],
                       "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split")
    expected_train = pd.DataFrame({"target": [1., 2.], "feat_cont": [2, 3], "feat_categ": [0, 1]})
    expected_validation = pd.DataFrame({"target": [3.], "feat_cont": [1], "feat_categ": [0]})
    expected_test = pd.DataFrame({"target": [4.], "feat_cont": [4], "feat_categ": [2]})
    pd.testing.assert_frame_equal(data_module.train, expected_train)
    pd.testing.assert_frame_equal(data_module.validation, expected_validation)
    pd.testing.assert_frame_equal(data_module.test, expected_test)
    assert data_module.categorical_columns == ["feat_categ"]
    assert data_module.numerical_columns == ["feat_cont"]
    assert data_module.categorical_dims == [4]
    check_is_fitted(data_module.scaler)


def test_wrong_data_types():
    current_date = datetime.today()
    df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_bool": [True, True, False, False],
                       "feat_date": [current_date, current_date, current_date, current_date],
                       "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split")
    expected_train = pd.DataFrame({"target": [0, 1], "feat_bool": [True, True],
                                   "feat_date": [current_date, current_date], "feat_categ": [0, 1]})
    expected_validation = pd.DataFrame({"target": [1], "feat_bool": [False],
                                        "feat_date": [current_date], "feat_categ": [0]})
    expected_test = pd.DataFrame({"target": [0], "feat_bool": [False],
                                  "feat_date": [current_date], "feat_categ": [2]})
    pd.testing.assert_frame_equal(data_module.train, expected_train)
    pd.testing.assert_frame_equal(data_module.validation, expected_validation)
    pd.testing.assert_frame_equal(data_module.test, expected_test)
    assert data_module.categorical_columns == ["feat_categ"]
    assert data_module.numerical_columns == []
    assert data_module.categorical_dims == [4]


def test_datamodule_no_categorical_columns():
    df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_cont": [2, 3, 1, 4],
                       "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split")
    expected_train = pd.DataFrame({"target": [0, 1], "feat_cont": [2, 3]})
    expected_validation = pd.DataFrame({"target": [1], "feat_cont": [1]})
    expected_test = pd.DataFrame({"target": [0], "feat_cont": [4]})
    pd.testing.assert_frame_equal(data_module.train, expected_train)
    pd.testing.assert_frame_equal(data_module.validation, expected_validation)
    pd.testing.assert_frame_equal(data_module.test, expected_test)
    assert data_module.categorical_columns == []
    assert data_module.numerical_columns == ["feat_cont"]
    # the target is always the last column
    assert data_module.categorical_dims == [1]
    check_is_fitted(data_module.scaler)


def test_datamodule_no_continuous_columns():
    df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_categ": ["a", "b", "a", "c"],
                       "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split")
    expected_train = pd.DataFrame({"target": [0, 1], "feat_categ": [0, 1]})
    expected_validation = pd.DataFrame({"target": [1], "feat_categ": [0]})
    expected_test = pd.DataFrame({"target": [0], "feat_categ": [2]})
    pd.testing.assert_frame_equal(data_module.train, expected_train)
    pd.testing.assert_frame_equal(data_module.validation, expected_validation)
    pd.testing.assert_frame_equal(data_module.test, expected_test)
    assert data_module.categorical_columns == ["feat_categ"]
    assert data_module.numerical_columns == []
    # the target is always the last column
    assert data_module.categorical_dims == [4]


def test_custom_scaler():
    df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_cont": [2, 3, 1, 4],
                       "feat_categ": ["a", "b", "a", "c"], "split": ["train", "train", "validation", "test"]})
    scaler = MinMaxScaler()
    data_module = SaintDatamodule(df=df, target="target", split_column="split", scaler=scaler)
    expected_train = pd.DataFrame({"target": [0, 1], "feat_cont": [2, 3], "feat_categ": [0, 1]})
    expected_validation = pd.DataFrame({"target": [1], "feat_cont": [1], "feat_categ": [0]})
    expected_test = pd.DataFrame({"target": [0], "feat_cont": [4], "feat_categ": [2]})
    pd.testing.assert_frame_equal(data_module.train, expected_train)
    pd.testing.assert_frame_equal(data_module.validation, expected_validation)
    pd.testing.assert_frame_equal(data_module.test, expected_test)
    assert data_module.categorical_columns == ["feat_categ"]
    assert data_module.numerical_columns == ["feat_cont"]
    # the target is always the last column
    assert data_module.categorical_dims == [4]
    check_is_fitted(data_module.scaler)
    assert data_module.scaler.__class__.__name__ == scaler.__class__.__name__


def test_fillnan():
    df = pd.DataFrame({"target": ["0", "1", "1", "0"], "feat_cont": [2, np.nan, 1, 4],
                       "feat_categ": ["a", "b", np.nan, "c"], "split": ["train", "train", "validation", "test"]})
    data_module = SaintDatamodule(df=df, target="target", split_column="split")
    expected_train = pd.DataFrame({"target": [0, 1], "feat_cont": [2., 0.], "feat_categ": [1, 2]})
    expected_validation = pd.DataFrame({"target": [1], "feat_cont": [1.], "feat_categ": [0]})
    expected_test = pd.DataFrame({"target": [0], "feat_cont": [4.], "feat_categ": [3]})
    pd.testing.assert_frame_equal(data_module.train, expected_train)
    pd.testing.assert_frame_equal(data_module.validation, expected_validation)
    pd.testing.assert_frame_equal(data_module.test, expected_test)
    assert data_module.categorical_columns == ["feat_categ"]
    assert data_module.numerical_columns == ["feat_cont"]
    # the target is always the last column
    assert data_module.categorical_dims == [5]
    check_is_fitted(data_module.scaler)
