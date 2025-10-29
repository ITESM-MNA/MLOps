"""
Pipeline orchestration for insurance project.
"""
import mlflow
from pathlib import Path
import pandas as pd
from src.load_data import DataLoader
from src.build_features import FeatureEngineer
from src.train_model import ModelTrainer
from src.evaluate_model import ModelEvaluator

class InsuranceMLflowPipeline:
    def __init__(self, config):
        self.config = config

    def run(self):
        mlflow.set_experiment(self.config['mlflow_experiment'])
        with mlflow.start_run():
            # 1. Carga de datos
            loader = DataLoader(**self.config['data']).load()
            # 2. Ingeniería de características
            fe = FeatureEngineer(loader.df_train, loader.df_test).add_features().select_features()
            # 3. Entrenamiento
            trainer = ModelTrainer(fe.df_train, fe.df_test, self.config['data']['target_col']).train()
            # 4. Evaluación
            evaluator = ModelEvaluator(trainer.models, fe.df_test, fe.df_test[self.config['data']['target_col']]).evaluate()
            # 5. Log de artefactos y métricas con MLflow
            for name, metrics in evaluator.items():
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{name}_{metric}", value)

if __name__ == "__main__":
    config = {
        "mlflow_experiment": "insurance_team42",
        "data": {
            "train_path": "data/ml_data/insurance_train.csv",
            "test_path": "data/ml_data/insurance_test.csv",
            "target_col": "MoHoPol"
        }
    }
    pipeline = InsuranceMLflowPipeline(config)
    pipeline.run()
