"""
Feature engineering utilities for insurance project.
"""
import pandas as pd

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

class FeatureEngineer:
    def __init__(self, df_train, df_test):
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.features = None

    def add_features(self):
        self.df_train = self._add_cross_features(self.df_train)
        self.df_test = self._add_cross_features(self.df_test)
        return self

    def select_features(self):
        features_base  = [f for f in PRIORITY_FEATURES if f in self.df_train.columns and f in self.df_test.columns]
        features_cross = [f for f in ["CAR_CROSS", "FIRE_CROSS"] if f in self.df_train.columns and f in self.df_test.columns]
        self.features = features_base + features_cross
        self.df_train = self.df_train[self.features + [col for col in self.df_train.columns if col not in self.features and col == 'MoHoPol']]
        self.df_test = self.df_test[self.features + [col for col in self.df_test.columns if col not in self.features and col == 'MoHoPol']]
        return self

    @staticmethod
    def _resolve_name(name, cols):
        if name in cols:
            return name
        alt = COIL_TO_CLEAN.get(name)
        if alt and alt in cols:
            return alt
        return None

    def _add_cross_features(self, df):
        c_auto = self._resolve_name("ContrCarPol", df.columns) or self._resolve_name("PPERSAUT", df.columns)
        n_auto = self._resolve_name("NumCarPol", df.columns)   or self._resolve_name("APERSAUT", df.columns)
        if c_auto and n_auto:
            df["CAR_CROSS"] = pd.to_numeric(df[c_auto], errors="coerce") * pd.to_numeric(df[n_auto], errors="coerce")
        c_fire = self._resolve_name("ContrFirePol", df.columns) or self._resolve_name("PBRAND", df.columns)
        n_fire = self._resolve_name("NumFirePol", df.columns)    or self._resolve_name("ABRAND", df.columns)
        if c_fire and n_fire:
            df["FIRE_CROSS"] = pd.to_numeric(df[c_fire], errors="coerce") * pd.to_numeric(df[n_fire], errors="coerce")
        return df
