"""
Model training and selection utilities for insurance project.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

RANDOM_STATE = 42

class ModelTrainer:
    def __init__(self, df_train, df_test, target_col):
        self.df_train = df_train
        self.df_test = df_test
        self.target_col = target_col
        self.models = {}
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def train(self):
        self.X_train = self.df_train.drop(columns=[self.target_col])
        self.y_train = self.df_train[self.target_col]
        self.X_test = self.df_test.drop(columns=[self.target_col])
        self.y_test = self.df_test[self.target_col]
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
        # Logistic Regression
        pipe_lr = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
        ])
        grid_lr = {"clf__C":[0.1,0.5,1.0,2.0], "clf__solver":["lbfgs"]}
        gs_lr = GridSearchCV(pipe_lr, grid_lr, cv=cv, scoring="average_precision", n_jobs=-1, verbose=0)
        gs_lr.fit(self.X_train, self.y_train)
        # Random Forest
        rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
        grid_rf = {"n_estimators": [200, 400], "max_depth": [None, 8, 12], "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]}
        gs_rf = GridSearchCV(rf, grid_rf, cv=cv, scoring="average_precision", n_jobs=-1, verbose=0)
        gs_rf.fit(self.X_train, self.y_train)
        # Gradient Boosting
        gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
        grid_gb = {"n_estimators":[200, 400], "max_depth": [3, 5], "learning_rate":[0.05, 0.1]}
        gs_gb = GridSearchCV(gb, grid_gb, cv=cv, scoring="average_precision", n_jobs=-1, verbose=0)
        gs_gb.fit(self.X_train, self.y_train)
        self.models = {
            "LogReg": gs_lr,
            "RandomForest": gs_rf,
            "GradBoost": gs_gb
        }
        return self
