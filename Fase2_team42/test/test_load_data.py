import pytest
import pandas as pd
from src.data.load_data import load_data

def test_load_data():
    train_path = "data/ml_data/insurance_train.csv"
    test_path = "data/ml_data/insurance_test.csv"
    target_col = "MoHoPol"
    df_train, df_test = load_data(train_path, test_path, target_col)
    assert not df_train.empty
    assert not df_test.empty
    assert target_col in df_train.columns
    assert target_col in df_test.columns
