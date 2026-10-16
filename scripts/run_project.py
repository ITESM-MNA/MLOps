"""Run the MLflow Project from Python instead of the CLI.

python scripts/run_project.py
"""

import mlflow

from mlops.config import MLFLOW_EXPERIMENT_NAME

submitted = mlflow.projects.run(
    uri=".",
    entry_point="main",
    parameters={"alpha": 0.3},
    env_manager="local",
    experiment_name=MLFLOW_EXPERIMENT_NAME,
)
print(submitted.run_id, submitted.get_status())
