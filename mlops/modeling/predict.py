"""Predict with a model logged to MLflow.

python -m mlops.modeling.predict --model-uri runs:/<run_id>/model
"""

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd

from mlops.config import PREDICTIONS_FILE, TARGET, TEST_FILE
from mlops.tracking import log_dataset, start_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True, help="e.g. runs:/<run_id>/model")
    parser.add_argument("--data", default=str(TEST_FILE))
    args = parser.parse_args()

    model = mlflow.sklearn.load_model(args.model_uri)

    df = pd.read_csv(args.data)
    input_df = df.copy()
    df["prediction"] = model.predict(df.drop(columns=[TARGET], errors="ignore"))

    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDICTIONS_FILE, index=False)

    with start_run():
        log_dataset(input_df, args.data, name="wine_quality_test", context="inference")
        mlflow.log_param("source_model", args.model_uri)
        mlflow.log_artifact(str(PREDICTIONS_FILE))

    print(f"Saved {PREDICTIONS_FILE.name}: {len(df)} rows")


if __name__ == "__main__":
    main()
