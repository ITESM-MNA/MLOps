import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,  make_scorer, classification_report,
    precision_recall_curve, roc_curve, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), '../../reports/')
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_train_test(data_dir, target_col):
    df_train = pd.read_csv(os.path.join(data_dir, "insurance_train.csv"))
    df_test  = pd.read_csv(os.path.join(data_dir, "insurance_test.csv"))
    for df in (df_train, df_test):
        if df[target_col].dtype != "int64" and df[target_col].dtype != "int32":
            df[target_col] = df[target_col].astype(int)
    return df_train, df_test

def get_features(df_train, df_test):
    PRIORITY_FEATURES = [
        "ContrCarPol", "NumCarPol", "ContrFirePol", "DemAvgIncome", "DemMidInc", "DemLoLeEdu", "DemHiLeEdu", "DemLowestInc", "ContrPrivIns", "CMainType",
    ]
    COIL_TO_CLEAN = {
        "PPERSAUT": "ContrCarPol",
        "APERSAUT": "NumCarPol",
        "PBRAND":   "ContrFirePol",
        "ABRAND":   "NumFirePol",
        "PWAPART":  "ContrPrivIns",
        "AWAPART":  "NumPrivIns",
        "MKOOPKLA": "CMainType",
    }
    def resolve_name(name, cols):
        if name in cols:
            return name
        alt = COIL_TO_CLEAN.get(name)
        if alt and alt in cols:
            return alt
        return None
    def add_cross(df):
        c_auto = resolve_name("ContrCarPol", df.columns) or resolve_name("PPERSAUT", df.columns)
        n_auto = resolve_name("NumCarPol", df.columns)   or resolve_name("APERSAUT", df.columns)
        if c_auto and n_auto:
            df["CAR_CROSS"] = pd.to_numeric(df[c_auto], errors="coerce") * pd.to_numeric(df[n_auto], errors="coerce")
        c_fire = resolve_name("ContrFirePol", df.columns) or resolve_name("PBRAND", df.columns)
        n_fire = resolve_name("NumFirePol", df.columns)    or resolve_name("ABRAND", df.columns)
        if c_fire and n_fire:
            df["FIRE_CROSS"] = pd.to_numeric(df[c_fire], errors="coerce") * pd.to_numeric(df[n_fire], errors="coerce")
        return df
    df_train_fx = add_cross(df_train.copy())
    df_test_fx  = add_cross(df_test.copy())
    features_base  = [f for f in PRIORITY_FEATURES if f in df_train_fx.columns and f in df_test_fx.columns]
    features_cross = [f for f in ["CAR_CROSS", "FIRE_CROSS"] if f in df_train_fx.columns and f in df_test_fx.columns]
    FEATURES = features_base + features_cross
    return df_train_fx, df_test_fx, FEATURES

def train_and_evaluate(X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    def summary_basic(name, y_true, y_score):
        return {
            "Modelo":  name,
            "ROC_AUC": roc_auc_score(y_true, y_score),
            "PR_AUC":  average_precision_score(y_true, y_score),
        }
    def summary_best_f1(name, y_true, y_score, steps=101):
        thrs = np.linspace(0.01, 0.99, steps)
        best = {"thr":0.5, "precision":0.0, "recall":0.0, "f1":0.0}
        for t in thrs:
            y_pred = (y_score >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best["f1"]:
                best = {
                    "thr": float(t),
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall":    recall_score(y_true, y_pred),
                    "f1":        f1,
                }
        return best
    # 1) Logistic Regression
    pipe_lr = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
    ])
    grid_lr = {"clf__C":[0.1,0.5,1.0,2.0], "clf__solver":["lbfgs"]}
    gs_lr = GridSearchCV(pipe_lr, grid_lr, cv=cv, scoring="average_precision", n_jobs=-1, verbose=0)
    gs_lr.fit(X_train, y_train)
    best_lr = gs_lr.best_estimator_
    proba_lr = best_lr.predict_proba(X_test)[:,1]
    # 2) Random Forest
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    grid_rf = {
        "n_estimators":     [200, 400],
        "max_depth":        [None, 8, 12],
        "min_samples_split":[2, 5],
        "min_samples_leaf": [1, 2],
    }
    gs_rf = GridSearchCV(rf, grid_rf, cv=cv, scoring="average_precision", n_jobs=-1, verbose=0)
    gs_rf.fit(X_train, y_train)
    best_rf = gs_rf.best_estimator_
    proba_rf = best_rf.predict_proba(X_test)[:,1]
    # 3) Gradient Boosting
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    grid_gb = {
        "n_estimators":[200, 400],
        "max_depth":   [3, 5],
        "learning_rate":[0.05, 0.1],
    }
    gs_gb = GridSearchCV(gb, grid_gb, cv=cv, scoring="average_precision", n_jobs=-1, verbose=0)
    gs_gb.fit(X_train, y_train)
    best_gb = gs_gb.best_estimator_
    proba_gb = best_gb.predict_proba(X_test)[:,1]
    # Comparación: AUC + mejor F1
    results = {}
    for name, proba in [
        ("LogReg",       proba_lr),
        ("RandomForest", proba_rf),
        ("GradBoost",    proba_gb),
    ]:
        results[name] = {
            **summary_basic(name, y_test, proba),
            **summary_best_f1(name, y_test, proba)
        }
    # Guardar resultados
    import json
    with open(os.path.join(REPORTS_DIR, "ml_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    # Guardar curvas ROC y PR
    from sklearn.metrics import roc_curve, precision_recall_curve
    for name, proba in [
        ("LogReg", proba_lr),
        ("RandomForest", proba_rf),
        ("GradBoost", proba_gb),
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y_test,proba):.3f}")
    plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend();
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curves.png")); plt.close()
    for name, proba in [
        ("LogReg", proba_lr),
        ("RandomForest", proba_rf),
        ("GradBoost", proba_gb),
    ]:
        prec, rec, _ = precision_recall_curve(y_test, proba)
        plt.plot(rec, prec, label=f"{name} AP={average_precision_score(y_test,proba):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall"); plt.legend();
    plt.savefig(os.path.join(REPORTS_DIR, "pr_curves.png")); plt.close()
    return results
