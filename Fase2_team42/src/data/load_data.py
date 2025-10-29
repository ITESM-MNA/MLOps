"""
Data loading utilities for insurance project.
"""
from pathlib import Path
import pandas as pd

def load_data(train_path: str, test_path: str, target_col: str):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    assert target_col in df_train.columns and target_col in df_test.columns, f"{target_col} not found."
    # Imputar nulos con la media antes de convertir a entero
    if df_train[target_col].isna().any():
        mean_val = df_train[target_col].mean()
        df_train[target_col] = df_train[target_col].fillna(mean_val)
    if df_test[target_col].isna().any():
        mean_val = df_test[target_col].mean()
        df_test[target_col] = df_test[target_col].fillna(mean_val)
    if df_train[target_col].dtype not in ["int64", "int32"]:
        df_train[target_col] = df_train[target_col].astype(int)
    if df_test[target_col].dtype not in ["int64", "int32"]:
        df_test[target_col] = df_test[target_col].astype(int)
    return df_train, df_test
