import os
import sys
import pandas as pd
from joblib import dump
from FaseFinal_team42.src.train_model import ModelTrainer

# Add project root directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Define paths
data_path = "c:/Users/mizlop/OneDrive - SAS/Documents/SAS_git/MLOps/FaseFinal_team42/data"
model_output_path = "c:/Users/mizlop/OneDrive - SAS/Documents/SAS_git/MLOps/FaseFinal_team42/models/final_model_GradBoost.joblib"

# Load data
df_train = pd.read_csv(os.path.join(data_path, "train.csv"))
df_test = pd.read_csv(os.path.join(data_path, "test.csv"))

# Train model
target_column = "target"  # Replace with the actual target column name
trainer = ModelTrainer(df_train, df_test, target_column)
trainer.train()

# Export Gradient Boosting model
best_gb_model = trainer.models["GradBoost"].best_estimator_
dump(best_gb_model, model_output_path)

print(f"Model exported to {model_output_path}")
