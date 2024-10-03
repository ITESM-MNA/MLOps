import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        # Define the class labels
        self.class_labels = ['Green frogs', 'Brown frogs', 'Common toad', 'Fire-bellied toad', 'Tree frog',
                             'Common newt', 'Great crested newt']

    def load_model(self):
        """
        Load only the model from the specified file path.
        """
        logger.info(f"Loading model from: {self.model_path}")
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)  # Only load the model
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise  # Re-raise the exception to stop execution

    def preprocess_input(self, input_data):
        """
        Preprocess the input data.
        In this case, it's already PCA transformed and scaled.
        """
        logger.info("Starting input data preprocessing...")
        return input_data

    def predict(self, input_data, best_thresholds=None):
        """
        Predict using the loaded model and apply the best thresholds for each label.
        """
        if self.model is None:
            logger.error("Model is not loaded. Please load the model before prediction.")
            return None

        logger.info("Starting prediction process...")
        preprocessed_data = self.preprocess_input(input_data)

        # Debugging: Check preprocessed input data
        logger.info(f"Preprocessed Input Data: {preprocessed_data}")

        # Get probabilities for each output (multi-output classification)
        probabilities = self.model.predict_proba(preprocessed_data)

        # Initialize a list to store predictions
        predictions = []

        # Apply thresholds (either the best thresholds or a default threshold of 0.5)
        for i, prob in enumerate(probabilities):
            # prob is a list of arrays, one array for each class; extract the second element (class 1)
            prob_class_1 = np.array([p[1] for p in prob])

            # Use the best threshold if provided, otherwise default to 0.5
            threshold = best_thresholds[i] if best_thresholds is not None else 0.5
            pred = (prob_class_1 >= threshold).astype(int)
            predictions.append(pred)

        # Stack predictions into a 2D array
        predictions = np.column_stack(predictions)

        # Debugging: Log final prediction result
        logger.info(f"Thresholded Prediction result: {predictions}")

        return predictions

    def decode_prediction(self, prediction):
        """
        Map prediction output to class labels.
        Assuming prediction is a binary matrix (e.g., 0 and 1 values), map it to human-readable labels.
        """
        decoded_prediction = []

        # Check if prediction length matches number of classes
        for pred in prediction:
            if len(pred) != len(self.class_labels):
                logger.error(
                    f"Prediction length {len(pred)} does not match the number of class labels {len(self.class_labels)}")
                return None  # Exit if there's a mismatch

            # Decode only when prediction length matches class labels
            decoded_classes = [self.class_labels[i] for i, val in enumerate(pred) if val == 1]
            decoded_prediction.append(decoded_classes)

        return decoded_prediction


