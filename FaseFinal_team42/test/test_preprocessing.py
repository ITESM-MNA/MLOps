import pytest
from src.ml_pipeline import load_train_test, get_features

@pytest.fixture
def sample_data_dir():
    return "data/ml_data"

@pytest.fixture
def target_column():
    return "MoHoPol"

def test_load_train_test(sample_data_dir, target_column):
    """Test loading of train and test datasets."""
    df_train, df_test = load_train_test(sample_data_dir, target_column)
    assert not df_train.empty
    assert not df_test.empty
    assert target_column in df_train.columns
    assert target_column in df_test.columns

def test_get_features(sample_data_dir, target_column):
    """Test feature extraction from datasets."""
    df_train, df_test = load_train_test(sample_data_dir, target_column)
    features_train, features_test = get_features(df_train, df_test)
    assert features_train is not None
    assert features_test is not None
    assert len(features_train) == len(df_train)
    assert len(features_test) == len(df_test)
