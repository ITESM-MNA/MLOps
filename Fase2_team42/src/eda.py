import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

REPORTS_DIR = os.path.join(os.path.dirname(__file__), '../../reports/')
os.makedirs(REPORTS_DIR, exist_ok=True)

class DataExplorer:
    @staticmethod
    def explore(data):
        print(data.head().T)
        print(data.describe())
        print(data.info())

    @staticmethod
    def plot_histograms(data):
        import matplotlib.pyplot as plt
        data.hist(bins=15, figsize=(15, 10))
        plt.show()

    @staticmethod
    def plot_correlations(data):
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.show()

def load_and_describe_data(input_path, column_names):
    df = pd.read_csv(input_path, header=None)
    df.columns = column_names
    df.to_csv(os.path.join(REPORTS_DIR, "insurance_df_head.csv"), index=False)
    info_path = os.path.join(REPORTS_DIR, "insurance_info.txt")
    with open(info_path, "w") as f:
        df.info(buf=f)
    desc_path = os.path.join(REPORTS_DIR, "insurance_describe.csv")
    df.describe().to_csv(desc_path)
    return df

def plot_target_distribution(df, target_col):
    plt.figure(figsize=(5,3))
    sns.countplot(x=target_col, data=df)
    plt.title(f'Distribución de la variable objetivo {target_col}')
    plt.savefig(os.path.join(REPORTS_DIR, f"target_distribution.png"))
    plt.close()

def save_top_correlations(df, target_col, top_n=10):
    corr = df.corr(numeric_only=True)[target_col].drop(target_col).abs()
    top = corr.sort_values(ascending=False).head(top_n)
    top.to_csv(os.path.join(REPORTS_DIR, "top10_correlations.csv"))
    return top

def plot_top_features(df, top_features, target_col):
    for feat in top_features:
        plt.figure(figsize=(6,4))
        sns.histplot(data=df, x=feat, hue=target_col, kde=True, element='step', stat='density', common_norm=False)
        plt.title(f"{feat} – Distribución por {target_col}")
        plt.savefig(os.path.join(REPORTS_DIR, f"hist_{feat}.png"))
        plt.close()

def split_and_save_train_test(df, target_col, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    train_df = X_train.copy()
    train_df[target_col] = y_train
    test_df = X_test.copy()
    test_df[target_col] = y_test
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/ml_data/')
    os.makedirs(data_dir, exist_ok=True)
    train_df.to_csv(os.path.join(data_dir, "insurance_train.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "insurance_test.csv"), index=False)
    return train_df, test_df
