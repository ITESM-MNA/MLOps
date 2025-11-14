import pytest
from src.load_data import DataLoader

def test_load_data():
    train_path = "data/ml_data/insurance_train.csv"
    test_path = "data/ml_data/insurance_test.csv"
    target_col = "MoHoPol"
    loader = DataLoader(train_path, test_path, target_col)
    loader.load()
    assert loader.df_train is not None
    assert loader.df_test is not None
    assert target_col in loader.df_train.columns
    assert target_col in loader.df_test.columns
