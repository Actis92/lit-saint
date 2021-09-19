import pandas as pd
from sklearn.utils.validation import check_is_fitted

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
    assert data_module.categorical_columns == [2]
    assert data_module.numerical_columns == [1]
    assert data_module.target_categorical
    # the target is always the last column
    assert data_module.categorical_dims == [3, 2]
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
    assert data_module.categorical_columns == [2]
    assert data_module.numerical_columns == [1]
    assert not data_module.target_categorical
    assert data_module.categorical_dims == [3]
    check_is_fitted(data_module.scaler)