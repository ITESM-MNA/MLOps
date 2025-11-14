"""
Evaluation and business metrics utilities for insurance project.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
)

class ModelEvaluator:
    def __init__(self, models, df_test, y_test):
        self.models = models
        self.df_test = df_test
        self.y_test = y_test
        self.results = {}

    def evaluate(self):
        X_test = self.df_test.drop(columns=[self.y_test.name]) if hasattr(self.y_test, 'name') else self.df_test
        y_test = self.y_test
        for name, gs in self.models.items():
            proba = gs.best_estimator_.predict_proba(X_test)[:, 1]
            self.results[name] = {
                "roc_auc": roc_auc_score(y_test, proba),
                "pr_auc": average_precision_score(y_test, proba),
                "f1": self._best_f1(y_test, proba)
            }
        return self.results

    @staticmethod
    def _best_f1(y_true, y_score, steps=101):
        thrs = np.linspace(0.01, 0.99, steps)
        best_f1 = 0
        for t in thrs:
            y_pred = (y_score >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
        return best_f1

def summary_basic(name, y_true, y_score):
    return pd.Series({
        "Modelo":  name,
        "ROC_AUC": roc_auc_score(y_true, y_score),
        "PR_AUC":  average_precision_score(y_true, y_score),
    })

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
    return pd.Series({
        "Mejor_thr(F1)": best["thr"],
        "precision*":   best["precision"],
        "recall*":      best["recall"],
        "f1*":          best["f1"],
    })

def business_report(y_true, y_score, thr, label="Top-X%"):
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    contactados = int((y_pred == 1).sum())
    total = len(y_true)
    hitrate = tp / max(1, contactados)
    capture = tp / max(1, (y_true == 1).sum())
    return pd.Series({
        "label": label,
        "threshold": float(thr),
        "contactados": contactados,
        "%contactados": round(100 * contactados / total, 2),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "hit_rate(=precision)": round(hitrate, 4),
        "capture(=recall)": round(capture, 4),
    })
