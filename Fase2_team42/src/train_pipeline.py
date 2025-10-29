"""
Pipeline orchestration for insurance project.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from data.load_data import load_data
from features.build_features import add_cross_features, select_features
from models.train_model import train_and_select_models
from models.evaluate_model import summary_basic, summary_best_f1, business_report
import mlflow
import mlflow.sklearn

DATA_DIR = Path("data/ml_data")
TRAIN_PATH = DATA_DIR / "insurance_train.csv"
TEST_PATH = DATA_DIR / "insurance_test.csv"
TARGET_COL = "MoHoPol"


def main():
    mlflow.set_experiment("insurance_team42")
    with mlflow.start_run():
        # Load data
        df_train, df_test = load_data(TRAIN_PATH, TEST_PATH, TARGET_COL)
        # Feature engineering
        df_train_fx = add_cross_features(df_train.copy())
        df_test_fx = add_cross_features(df_test.copy())
        features = select_features(df_train_fx, df_test_fx)
        X_train = df_train_fx[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_train = df_train_fx[TARGET_COL].astype(int)
        X_test = df_test_fx[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_test = df_test_fx[TARGET_COL].astype(int)
        # Log data shape
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        # Train models
        models = train_and_select_models(X_train, y_train)
        # Evaluate
        results = {}
        for name, gs in models.items():
            proba = gs.best_estimator_.predict_proba(X_test)[:,1]
            auc = summary_basic(name, y_test, proba)["ROC_AUC"]
            prauc = summary_basic(name, y_test, proba)["PR_AUC"]
            mlflow.log_metric(f"{name}_roc_auc", auc)
            mlflow.log_metric(f"{name}_pr_auc", prauc)
            # Log best model
            if name == "GradBoost":
                mlflow.sklearn.log_model(gs.best_estimator_, "model_gradboost")
            elif name == "RandomForest":
                mlflow.sklearn.log_model(gs.best_estimator_, "model_rf")
            elif name == "LogReg":
                mlflow.sklearn.log_model(gs.best_estimator_, "model_logreg")
            results[name] = {
                "summary": summary_basic(name, y_test, proba),
                "best_f1": summary_best_f1(name, y_test, proba)
            }
        # Print results
        for name, res in results.items():
            print(f"Model: {name}")
            print(res["summary"])
            print(res["best_f1"])

if __name__ == "__main__":
    main()
