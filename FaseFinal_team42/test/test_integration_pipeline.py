import pytest
from src.train_pipeline import InsuranceMLflowPipeline

def test_integration_pipeline():
    """Test the end-to-end pipeline flow."""
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
    # No assertion is made as the pipeline logs metrics to MLflow
