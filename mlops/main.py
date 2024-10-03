import argparse
import gin
import logging
import pickle
import random
import numpy as np
from mlops.dataset.data_loader import DataLoader
from mlops.feature_engineering.data_preprocessing import DataPreprocessing
from mlops.modeling.train_model import TrainModel
from mlops.modeling.prediction import Predictor

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

def run():
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
    with open(f'{best_model_name.lower()}_model.pkl', 'wb') as f:
        pickle.dump(trained_model, f)
    logger.info(f"Trained model saved at {best_model_name.lower()}_model.pkl")

    # Initialize the Predictor class
    predictor = Predictor(model_path=f'{best_model_name.lower()}_model.pkl')

    # Load the trained model
    predictor.load_model()

    # Evaluate the model on the test set
    precision, recall, f1, accuracy, hamming = model_trainer.evaluate_model_performance(trained_model)

    # Pick more samples from the dataset and run predictions
    for sample_idx in [0, 1, 5, 10]:  # You can adjust these indices as needed
        random_sample = model_trainer.X_test[sample_idx].reshape(1, -1)  # Shape the sample correctly for prediction
        actual_label = model_trainer.y_test[sample_idx]  # Get the actual label for this sample

        logger.info(f"Random sample for prediction: {random_sample}")
        logger.info(f"Actual label for the sample: {actual_label}")

        # Decode the actual label
        decoded_actual_label = predictor.decode_prediction(
            [actual_label.tolist()])  # Ensure it's a list of lists for decoding

        # Make the prediction
        prediction = predictor.predict(random_sample)

        logger.info(f"Raw Prediction result: {prediction}")

        # Decode the prediction to get class names
        decoded_prediction = predictor.decode_prediction(prediction)

        logger.info(f"Predicted classes: {decoded_prediction}")
        logger.info(f"Actual classes: {decoded_actual_label}")

        # Handle the case where both predicted and actual labels are empty
        if not decoded_prediction and not decoded_actual_label:
            logger.info("Both predicted and actual classes are empty — they match!")
        elif decoded_prediction == decoded_actual_label:
            logger.info("Prediction matches the actual label!")
        else:
            logger.info("Prediction does NOT match the actual label.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    gin.parse_config_file(args.config)
    run()
