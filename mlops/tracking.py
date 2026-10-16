"""Helpers for MLflow dataset tracking (https://mlflow.org/docs/latest/ml/dataset/).

Every step of the MLflow project logs the dataset it reads with ``mlflow.log_input``, so each
run records the name, digest, schema and source of its data.
"""

import os
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
import pandas as pd

from mlops.config import MLFLOW_EXPERIMENT_NAME, PROJ_ROOT, TARGET


def start_run():
    """Start (or resume) the MLflow run for a project step.

    Under `mlflow run` the run already exists (MLFLOW_RUN_ID is set) and is reused. When a step is
    executed directly with `python -m`, its run goes to the project's experiment.
    """
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return mlflow.start_run()


def artifact_source(run_id: str, artifact_path: str) -> str:
    """Location of an artifact logged by a previous run (usable as a dataset source)."""
    return f"{mlflow.get_run(run_id).info.artifact_uri}/{artifact_path}"


def dataset_source(path) -> str:
    """Return a portable source string: relative to the project root when possible."""
    path_str = str(path)
    if ":/" in path_str and not Path(path_str).drive:  # a URI such as s3://... or file:///...
        return path_str
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJ_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def log_dataset(df: pd.DataFrame, source: str, name: str, context: str) -> None:
    """Log ``df`` as an input of the active MLflow run."""
    targets = TARGET if TARGET in df.columns else None
    try:
        dataset = mlflow.data.from_pandas(
            df, source=dataset_source(source), name=name, targets=targets
        )
    except MlflowException:
        # MLflow cannot interpret this kind of source: log the dataset (digest, schema) without it
        dataset = mlflow.data.from_pandas(df, name=name, targets=targets)
    mlflow.log_input(dataset, context=context)
