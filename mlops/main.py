import argparse
import logging
import pickle
import os
import gin
import numpy as np
from sklearnex import patch_sklearn

from feature_engineering.data_preprocessing import DataPreprocessing
from modeling.prediction import Predictor
from modeling.train_model import TrainModel

patch_sklearn()


# Function to configure logging to both console and file
@gin.configurable
def configure_logging(reports_dir):

    # Ensure the reports directory exists
    os.makedirs(reports_dir, exist_ok=True)

    # Path to the log file inside reports_dir
    log_file_path = os.path.join(reports_dir, 'pipeline.log')

    # Set up logging with both console and file handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(),  # Logs to console
            logging.FileHandler(log_file_path, mode='w')  # Logs to file
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Logs will be saved to {log_file_path}")
    return logger

def run():
    try:
        # Load and preprocess the data
        X_pca, y = DataPreprocessing.load_and_preprocess_data()

        # Train the best model and save it
        train_model = TrainModel(X_pca, y)
        best_model_name, model_trainer = train_model.train_and_save_best_model()

        # Load the trained model
        predictor = Predictor(model_path=f'models/{best_model_name.lower()}_model.pkl')
        predictor.load_model()

        # Evaluate model performance
        mean_metrics, class_metrics = model_trainer.evaluate_model_performance(predictor.model)

        # Log model evaluation metrics
        logger.info(f"Evaluation metrics: {mean_metrics}")

        # Run predictions on a few samples
        sample_indices = [0, 1, 5, 10]  # You can adjust these indices as needed
        Predictor.run_predictions(predictor, model_trainer, sample_indices)

    except Exception as e:
        logger.error(f"An error occurred in the pipeline: {e}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    # Load Gin configuration
    try:
        gin.parse_config_file(args.config)
        logger = configure_logging()  # Move logging configuration after gin config is loaded
        logger.info("Gin configuration loaded successfully.")
    except Exception as e:
        print(f"Failed to load Gin configuration: {e}")
        raise

    # Execute the main run function
    run()
