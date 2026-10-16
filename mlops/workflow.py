"""Multi-step workflow: load -> features -> train.

Each step is its own MLflow run and passes its run_id to the next one.

    mlflow run . -e workflow --env-manager local --experiment-name wine-quality
    python -m mlops.workflow
"""

import mlflow
from mlflow.tracking import MlflowClient

from mlops.tracking import start_run

ENV = "local"  # reuse the current environment for every step


def main() -> None:
    with start_run() as parent:
        load = mlflow.run(".", "load", env_manager=ENV)
        feats = mlflow.run(
            ".",
            "features",
            env_manager=ENV,
            parameters={"run_id": load.run_id},
        )
        train = mlflow.run(
            ".",
            "train",
            env_manager=ENV,
            parameters={"run_id": feats.run_id},
        )

        # Optional: group the 3 runs under the parent in the UI.
        client = MlflowClient()
        for step in (load, feats, train):
            client.set_tag(step.run_id, "mlflow.parentRunId", parent.info.run_id)

    print(f"Workflow finished. Model URI: runs:/{train.run_id}/model")


if __name__ == "__main__":
    main()
