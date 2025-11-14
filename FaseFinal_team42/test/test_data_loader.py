import pytest
from src.load_data import DataLoader

@pytest.fixture
def data_loader():
    train_path = "data/ml_data/insurance_train.csv"
    test_path = "data/ml_data/insurance_test.csv"
    target_col = "MoHoPol"
    return DataLoader(train_path, test_path, target_col)

def test_load_data(data_loader):
    """Test loading of data and target column validation."""
    data_loader.load()
    assert data_loader.df_train is not None
    assert data_loader.df_test is not None
    assert data_loader.target_col in data_loader.df_train.columns
    assert data_loader.target_col in data_loader.df_test.columns

def test_target_column_type(data_loader):
    """Test that the target column is converted to integer."""
    data_loader.load()
    assert data_loader.df_train[data_loader.target_col].dtype == int
    assert data_loader.df_test[data_loader.target_col].dtype == int

def test_no_missing_values_in_target(data_loader):
    """Test that there are no missing values in the target column."""
    data_loader.load()
    assert not data_loader.df_train[data_loader.target_col].isna().any()
    assert not data_loader.df_test[data_loader.target_col].isna().any()
