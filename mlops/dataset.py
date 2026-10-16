"""Step 1 (load): cleaned dataset -> data/interim.

Reads the cleaned white-wine dataset produced by notebook 1.4
(data/processed/wine_quality_cleaned_dropna.csv) and prepares it for the MLflow project.

    python -m mlops.dataset --data data/processed/wine_quality_cleaned_dropna.csv
"""

import argparse

import mlflow
import pandas as pd

from mlops.config import INTERIM_FILE, SOURCE_DATA_FILE
from mlops.tracking import log_dataset, start_run


def load_data(path) -> pd.DataFrame:
    """Read a comma-separated CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Use snake_case column names and drop missing values and duplicates."""
    out = df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
    return out.dropna().drop_duplicates().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(SOURCE_DATA_FILE))
    args = parser.parse_args()

    source_df = load_data(args.data)
    df = clean_data(source_df)

    INTERIM_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM_FILE, index=False)

    # No arguments: reuses the run opened by `mlflow run` (or creates one).
    with start_run():
        log_dataset(source_df, args.data, name="wine_quality_cleaned_dropna", context="source")
        mlflow.log_param("source_path", args.data)
        mlflow.log_metric("rows", len(df))
        mlflow.log_artifact(str(INTERIM_FILE))  # the next step downloads it

    print(f"Saved {INTERIM_FILE.name}: {len(df)} rows")


if __name__ == "__main__":
    main()
