from .load_data import DataLoader
from .build_features import FeatureEngineer
from .train_model import ModelTrainer
from .evaluate_model import ModelEvaluator
from .train_pipeline import InsuranceMLflowPipeline

# Punto de entrada para ejecutar el pipeline completo desde el paquete src

def run_pipeline():
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
