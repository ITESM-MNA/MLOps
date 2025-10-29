"""
Preprocesamiento y EDA reproducible para CoIL 2000 (Team 42)
"""
import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from pathlib import Path

# Rutas multiplataforma
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "insurance_company_modified.csv"
DICT_PATH = DATA_DIR / "TicDataDescr_newname.txt"
REPORTS_DIR = BASE_DIR / "reports"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

# 1. Carga robusta del CSV
na_tokens = ["", " ", "NA", "N/A", "NULL", "null", "None", "-", "?", "—", "nan", "NaN"]
df = pd.read_csv(RAW_PATH, na_values=na_tokens, keep_default_na=True)
print("Dimensiones iniciales:", df.shape)

# 2. Parseo tolerante del diccionario
def parse_dictionary(dict_path):
    numbers, names, descriptions, domains, new_names = [], [], [], [], []
    in_table = False
    with open(dict_path, "r", encoding="latin-1") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("DATADICTIONARY"):
                in_table = True
                continue
            if not in_table:
                continue
            if line.startswith("L0:"):
                break
            if line.startswith("Nr "):
                continue
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                num = int(parts[0])
                name = parts[1]
                new_name = parts[2]
                rest = line[len(parts[0])+len(parts[1])+len(parts[2])+3:].strip()
                dom = ""
                desc = rest
                see_idx = rest.lower().rfind(" see ")
                if see_idx != -1:
                    desc = rest[:see_idx].strip()
                    dom = rest[see_idx+1:].strip()
                else:
                    tokens = rest.split()
                    dom_start = None
                    for i, t in enumerate(tokens):
                        if re.search(r"(see|[0-9%–\-f])", t, flags=re.I):
                            dom_start = i
                            break
                    if dom_start is not None:
                        desc = " ".join(tokens[:dom_start]).strip()
                        dom = " ".join(tokens[dom_start:]).strip()
                numbers.append(num)
                names.append(name)
                new_names.append(new_name)
                descriptions.append(desc)
                domains.append(dom)
    return pd.DataFrame({
        "Column": numbers,
        "Name": names,
        "NewName": new_names,
        "Description": descriptions,
        "Domain": domains
    }).sort_values("Column")

data_dict = parse_dictionary(DICT_PATH)
data_dict.to_csv(REPORTS_DIR / "data_dictionary.csv", index=False, encoding="utf-8")
print("Diccionario cargado:", data_dict.shape)

# Validación columnas
missing_in_csv = [c for c in data_dict["NewName"] if c not in df.columns]
extra_in_csv = [c for c in df.columns if c not in set(data_dict["NewName"])]
print("Columnas del diccionario NO presentes en el CSV:", len(missing_in_csv))
print(missing_in_csv[:20])
print("Columnas del CSV NO documentadas en diccionario:", len(extra_in_csv))
print(extra_in_csv[:20])

# 3. Normalización de formatos y coerción a numérico
def coerce_numeric(df):
    coercion_report = {}
    for col in df.select_dtypes(include=["object"]).columns:
        before = df[col].isna().sum()
        df[col] = (df[col].astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.strip())
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after = df[col].isna().sum()
        if after != before:
            coercion_report[col] = {"nulos_antes": int(before), "nulos_despues": int(after)}
    return df, coercion_report

df, coercion_report = coerce_numeric(df)
with open(REPORTS_DIR / "coercion_report.json", "w", encoding="utf-8") as f:
    json.dump(coercion_report, f, indent=2)

# 4. Imputación de valores faltantes (ejemplo genérico)
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isna().any():
        median = df[col].median()
        df[col] = df[col].fillna(median)
for col in df.select_dtypes(include=["object"]).columns:
    if df[col].isna().any():
        mode = df[col].mode().iloc[0] if not df[col].mode().empty else ""
        df[col] = df[col].fillna(mode)

# 5. Tratamiento de outliers (IQR capping)
def iqr_cap(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return s.clip(lower, upper)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = iqr_cap(df[col])

# 6. Features derivados (ejemplo: CAR_CROSS, FIRE_CROSS)
if "PPERSAUT" in df.columns and "APERSAUT" in df.columns:
    df["CAR_CROSS"] = df["PPERSAUT"] * df["APERSAUT"]
if "PBRAND" in df.columns and "ABRAND" in df.columns:
    df["FIRE_CROSS"] = df["PBRAND"] * df["ABRAND"]

# 7. Guardado del dataset limpio
clean_path = PROCESSED_DIR / "insurance_clean.csv"
df.to_csv(clean_path, index=False)

# 8. Reporte de calidad
quality_report = {
    "shape": df.shape,
    "nulos_totales": int(df.isna().sum().sum()),
    "columnas_con_nulos": df.columns[df.isna().any()].tolist(),
    "data_dictionary": str(REPORTS_DIR / "data_dictionary.csv"),
    "coercion_report": str(REPORTS_DIR / "coercion_report.json"),
    "dataset_limpio": str(clean_path)
}
with open(REPORTS_DIR / "quality_report.json", "w", encoding="utf-8") as f:
    json.dump(quality_report, f, indent=2)

print("Preprocesamiento y EDA completados. Dataset limpio y reportes generados.")
