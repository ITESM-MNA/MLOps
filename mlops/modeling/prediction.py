import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_path, preprocessor_path=None, pca_path=None):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.pca_path = pca_path
        self.model = None
        self.preprocessor = None
        self.pca = None
        self.class_labels = ['Green frogs', 'Brown frogs', 'Common toad', 'Fire-bellied toad', 'Tree frog',
                             'Common newt', 'Great crested newt']

    def load_model(self):
        logger.info(f"Loading model from: {self.model_path}")
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def load_preprocessor(self):
        if self.preprocessor_path:
            logger.info(f"Loading preprocessor from: {self.preprocessor_path}")
            try:
                with open(self.preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
                logger.info("Preprocessor loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading preprocessor: {e}")
                raise
        else:
            logger.warning("No preprocessor path provided. Skipping preprocessor loading.")

    def load_pca(self):
        if self.pca_path:
            logger.info(f"Loading PCA from: {self.pca_path}")
            try:
                with open(self.pca_path, 'rb') as f:
                    self.pca = pickle.load(f)
                logger.info("PCA loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading PCA: {e}")
                raise
        else:
            logger.warning("No PCA path provided. Skipping PCA loading.")

    def load_all(self):
        self.load_model()
        self.load_preprocessor()
        self.load_pca()

    def preprocess_input(self, input_data):
        logger.info("Starting input data preprocessing...")

        if isinstance(input_data, pd.DataFrame):
            # If input is a DataFrame, use column names
            input_data = input_data.to_numpy()
        elif not isinstance(input_data, np.ndarray):
            raise ValueError("Input data must be a pandas DataFrame or a numpy array")

        if self.preprocessor:
            input_data = self.preprocessor.transform(input_data)
            logger.info("Applied preprocessor transformation.")
        else:
            logger.warning("No preprocessor available. Skipping preprocessing step.")

        if self.pca:
            input_data = self.pca.transform(input_data)
            logger.info("Applied PCA transformation.")
        else:
            logger.warning("No PCA model available. Skipping PCA transformation.")

        return input_data

    def predict(self, input_data, best_thresholds=None):
        if self.model is None:
            logger.error("Model is not loaded. Please load the model before prediction.")
            return None

        logger.info("Starting prediction process...")
        preprocessed_data = self.preprocess_input(input_data)

        logger.info(f"Preprocessed Input Data shape: {preprocessed_data.shape}")

        probabilities = self.model.predict_proba(preprocessed_data)

        predictions = []
        for i, prob in enumerate(probabilities):
            prob_class_1 = prob[:, 1] if prob.ndim == 2 else prob
            threshold = best_thresholds[i] if best_thresholds is not None else 0.5
            pred = (prob_class_1 >= threshold).astype(int)
            predictions.append(pred)

        predictions = np.column_stack(predictions)

        logger.info(f"Prediction shape: {predictions.shape}")
        logger.info(f"Sample of predictions: {predictions[:5]}")

        return predictions

    def decode_prediction(self, prediction):
        decoded_prediction = []

        if prediction.shape[1] != len(self.class_labels):
            logger.error(
                f"Prediction shape {prediction.shape} does not match the number of class labels {len(self.class_labels)}")
            return None

        for pred in prediction:
            decoded_classes = [self.class_labels[i] for i, val in enumerate(pred) if val == 1]
            decoded_prediction.append(decoded_classes)

        return decoded_prediction

    def predict_and_decode(self, input_data, best_thresholds=None):
        predictions = self.predict(input_data, best_thresholds)
        return self.decode_prediction(predictions)

    def run_predictions(predictor, model_trainer, sample_indices):
        try:
            for sample_idx in sample_indices:
                random_sample = model_trainer.X_test[sample_idx].reshape(1, -1)
                actual_label = model_trainer.y_test[sample_idx]

                logger.info(f"Random sample for prediction: {random_sample}")
                logger.info(f"Actual label for the sample: {actual_label}")

                # Convert actual_label to a numpy array before decoding
                actual_label_np = np.array([actual_label])

                # Decode the actual label
                decoded_actual_label = predictor.decode_prediction(actual_label_np)

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
