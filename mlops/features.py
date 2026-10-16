"""Step 2 (features): data/interim -> data/processed.

python -m mlops.features --run-id <run_id of the load step>
"""

import argparse

import mlflow
import pandas as pd

from mlops.config import FEATURES_FILE, INTERIM_FILE
from mlops.tracking import artifact_source, log_dataset, start_run


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["total_acidity"] = out["fixed_acidity"] + out["volatile_acidity"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="run_id of the load step")
    args = parser.parse_args()

    # Download the previous step's output from MLflow Tracking.
    interim_path = mlflow.artifacts.download_artifacts(
        run_id=args.run_id, artifact_path=INTERIM_FILE.name
    )
    interim_df = pd.read_csv(interim_path)
    df = build_features(interim_df)

    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_FILE, index=False)

    with start_run():
        log_dataset(
            interim_df,
            artifact_source(args.run_id, INTERIM_FILE.name),
            name="wine_quality_interim",
            context="interim",
        )
        mlflow.log_param("load_run_id", args.run_id)
        mlflow.log_artifact(str(FEATURES_FILE))

    print(f"Saved {FEATURES_FILE.name}: {df.shape}")


if __name__ == "__main__":
    main()
