import argparse
import logging
import pickle
import random

import gin
import numpy as np
from sklearnex import patch_sklearn

from mlops.dataset.data_loader import DataLoader
from mlops.feature_engineering.data_preprocessing import DataPreprocessing
from mlops.modeling.prediction import Predictor
from mlops.modeling.train_model import TrainModel

patch_sklearn()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def load_and_preprocess_data():
    try:
        # Initialize the DataLoader
        data_loader = DataLoader()

        # Load the dataset
        df = data_loader.load()

        # Preprocess the data and labels
        data_preprocessor = DataPreprocessing(df)
        X_pca, y = data_preprocessor.preprocess()  # y is already binarized and ready for use

        # Log the initial shapes of X_pca and y
        logger.info(f"Shape of X_pca: {X_pca.shape}")
        logger.info(f"Shape of y: {y.shape}")
        return X_pca, y
    except Exception as e:
        logger.error(f"Error during data loading and preprocessing: {e}")
        raise


def train_and_save_best_model(X_pca, y):
    try:
        # Train and test the model
        model_trainer = TrainModel(X_pca, y)
        model_trainer.train_test_split()

        # Train all models and get the results
        results = model_trainer.run_all_models()

        # Find the model with the best F1 score
        best_model_name = max(results, key=lambda x: results[x]['f1_score'])
        logger.info(f"Best model based on F1 Score: {best_model_name}")

        # Train the best model
        trained_model = model_trainer.train_model(best_model_name)

        # Save the trained model
        model_path = f'{best_model_name.lower()}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(trained_model, f)
        logger.info(f"Trained model saved at {model_path}")

        # Save best thresholds
        with open(f'{best_model_name.lower()}_thresholds.pkl', 'wb') as f:
            pickle.dump(model_trainer.best_thresholds, f)
        logger.info(f"Best thresholds saved at {best_model_name.lower()}_thresholds.pkl")

        return best_model_name, model_trainer
    except Exception as e:
        logger.error(f"Error during model training or saving: {e}")
        raise


def run_predictions(predictor, model_trainer, sample_indices):
    try:
        for sample_idx in sample_indices:
            random_sample = model_trainer.X_test[sample_idx].reshape(1, -1)
            actual_label = model_trainer.y_test[sample_idx]

            logger.info(f"Random sample for prediction: {random_sample}")
            logger.info(f"Actual label for the sample: {actual_label}")

            # Decode the actual label
            decoded_actual_label = predictor.decode_prediction([actual_label.tolist()])

            # Make the prediction
            raw_prediction = predictor.predict(random_sample)

            # Apply the best thresholds to get final class predictions
            thresholded_prediction = (raw_prediction >= model_trainer.best_thresholds).astype(int)

            # Decode the thresholded prediction
            decoded_prediction = predictor.decode_prediction(thresholded_prediction)

            logger.info(f"Thresholded Prediction result: {thresholded_prediction}")
            logger.info(f"Raw Prediction result: {raw_prediction}")
            logger.info(f"Predicted classes: {decoded_prediction}")
            logger.info(f"Actual classes: {decoded_actual_label}")

            # Check if the prediction matches the actual label
            if (thresholded_prediction == actual_label).all():
                logger.info("Prediction matches the actual label!")
            else:
                logger.info("Prediction does NOT match the actual label.")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


def run():
    try:
        # Load and preprocess the data
        X_pca, y = load_and_preprocess_data()

        # Train the best model and save it
        best_model_name, model_trainer = train_and_save_best_model(X_pca, y)

        # Load the trained model
        predictor = Predictor(model_path=f'{best_model_name.lower()}_model.pkl')
        predictor.load_model()

        # Evaluate model performance
        mean_metrics, class_metrics = model_trainer.evaluate_model_performance(predictor.model)

        # Log model evaluation metrics
        logger.info(f"Evaluation metrics: {mean_metrics}")

        # Run predictions on a few samples
        sample_indices = [0, 1, 5, 10]  # You can adjust these indices as needed
        run_predictions(predictor, model_trainer, sample_indices)

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
        logger.info("Gin configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Gin configuration: {e}")
        raise

    # Execute the main run function
    run()
