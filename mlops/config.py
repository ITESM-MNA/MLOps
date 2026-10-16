import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Dataset storage: datasets in data/processed are versioned with MLflow dataset tracking
# (https://mlflow.org/docs/latest/ml/dataset/). No DVC and no cloud bucket are used.
DATASET_STORAGE_DIR = PROCESSED_DATA_DIR

# MLflow tracking server: the locally running instance started from the project root with
#   mlflow server --host 127.0.0.1 --port 5000
# Override it by setting MLFLOW_TRACKING_URI in the .env file (e.g. to use a remote server).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
# Export it so every MLflow call in this process and its subprocesses (mlflow.run) uses the server
os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Wine-quality MLflow project (MLproject entry points and notebooks 2.x).
# It starts from the cleaned white-wine dataset produced by notebook 1.4.
TARGET = "quality"
RANDOM_STATE = 42
SOURCE_DATA_FILE = PROCESSED_DATA_DIR / "wine_quality_cleaned_dropna.csv"
INTERIM_FILE = INTERIM_DATA_DIR / "wine_quality_interim.csv"
FEATURES_FILE = PROCESSED_DATA_DIR / "wine_quality_features.parquet"
TRAIN_FILE = PROCESSED_DATA_DIR / "wine_quality_train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "wine_quality_test.csv"
PREDICTIONS_FILE = REPORTS_DIR / "predictions.csv"

# Experiment used by the MLflow project (override with MLFLOW_EXPERIMENT_NAME in .env)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "wine-quality")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
