"""Step 3 (train): fit the model and log it to MLflow.

python -m mlops.modeling.train --data data/processed/wine_quality_train.csv --alpha 0.3
python -m mlops.modeling.train --run-id <run_id of the features step>
"""

import argparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops.config import FEATURES_FILE, RANDOM_STATE, TARGET, TRAIN_FILE
from mlops.tracking import artifact_source, log_dataset, start_run
from mlops.transformers import OutlierClipper


def load_data(data: str, run_id: str | None) -> tuple[pd.DataFrame, str]:
    """Read a local CSV, or download the features file from a previous run."""
    if run_id:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=FEATURES_FILE.name)
        return pd.read_parquet(path), artifact_source(run_id, FEATURES_FILE.name)
    return pd.read_csv(data), data


def build_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("clip", OutlierClipper(q=0.99)),
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=alpha, random_state=RANDOM_STATE)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(TRAIN_FILE))
    parser.add_argument("--run-id", default=None, help="run_id of the features step")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    df, source = load_data(args.data, args.run_id)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline(args.alpha).fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }

    # No arguments: reuses the run opened by `mlflow run` (or creates one with `python -m`).
    with start_run() as run:
        log_dataset(
            X_train.assign(**{TARGET: y_train}),
            source,
            name="wine_quality_train",
            context="training",
        )
        log_dataset(
            X_test.assign(**{TARGET: y_test}),
            source,
            name="wine_quality_validation",
            context="validation",
        )
        mlflow.log_params({"alpha": args.alpha, "data_source": source})
        mlflow.log_metrics(metrics)
        # cloudpickle: the default format (skops) cannot serialize custom classes.
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )

    print(f"alpha={args.alpha} " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    print(f"Model URI: runs:/{run.info.run_id}/model")


if __name__ == "__main__":
    main()
