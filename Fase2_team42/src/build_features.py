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

def resolve_name(name: str, cols):
    if name in cols:
        return name
    alt = COIL_TO_CLEAN.get(name)
    if alt and alt in cols:
        return alt
    return None

def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    c_auto = resolve_name("ContrCarPol", df.columns) or resolve_name("PPERSAUT", df.columns)
    n_auto = resolve_name("NumCarPol", df.columns)   or resolve_name("APERSAUT", df.columns)
    if c_auto and n_auto:
        df["CAR_CROSS"] = pd.to_numeric(df[c_auto], errors="coerce") * pd.to_numeric(df[n_auto], errors="coerce")
    c_fire = resolve_name("ContrFirePol", df.columns) or resolve_name("PBRAND", df.columns)
    n_fire = resolve_name("NumFirePol", df.columns)    or resolve_name("ABRAND", df.columns)
    if c_fire and n_fire:
        df["FIRE_CROSS"] = pd.to_numeric(df[c_fire], errors="coerce") * pd.to_numeric(df[n_fire], errors="coerce")
    return df

def select_features(df_train, df_test):
    features_base  = [f for f in PRIORITY_FEATURES if f in df_train.columns and f in df_test.columns]
    features_cross = [f for f in ["CAR_CROSS", "FIRE_CROSS"] if f in df_train.columns and f in df_test.columns]
    return features_base + features_cross
