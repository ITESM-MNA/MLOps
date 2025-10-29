"""
Data loading utilities for insurance project.
"""
from pathlib import Path
import pandas as pd

class DataLoader:
    def __init__(self, train_path, test_path, target_col):
        self.train_path = train_path
        self.test_path = test_path
        self.target_col = target_col
        self.df_train = None
        self.df_test = None

    def load(self):
        self.df_train = pd.read_csv(self.train_path)
        self.df_test = pd.read_csv(self.test_path)
        assert self.target_col in self.df_train.columns and self.target_col in self.df_test.columns, f"{self.target_col} not found."
        # Imputar nulos con la media antes de convertir a entero
        for df in [self.df_train, self.df_test]:
            if df[self.target_col].isna().any():
                mean_val = df[self.target_col].mean()
                df[self.target_col] = df[self.target_col].fillna(mean_val)
            if df[self.target_col].dtype not in ["int64", "int32"]:
                df[self.target_col] = df[self.target_col].astype(int)
        return self
