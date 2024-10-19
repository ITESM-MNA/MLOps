import logging
import numpy as np
import os
import pickle
import gin
import importlib
import mlflow
import yaml

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score, hamming_loss,
                             precision_recall_curve, make_scorer)
from sklearn.preprocessing import StandardScaler


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TrainModel:
    def __init__(self, X, y, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_df = None
        self.y_test_df = None
        self.test_size = self.config['training']['test_size']
        self.random_state = self.config['training']['random_state']
        self.cv_folds = self.config['training']['cv_folds']
        self.best_thresholds = None
        self.models = self.load_models()
        self.hyperparameters = self.load_hyperparameters()

    def load_models(self):
        models = {}
        for model_name, model_config in self.config['models'].items():
            module_name, class_name = model_config['class'].rsplit('.', 1)
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            models[model_name] = MultiOutputClassifier(model_class())
        return models

    def load_hyperparameters(self):
        hyperparameters = {}
        for model_name, model_config in self.config['models'].items():
            hyperparameters[model_name] = {
                f'estimator__{k}': v for k, v in model_config['hyperparameters'].items()
            }
        return hyperparameters

    def train_test_split(self):
        """
        Split the data into training and testing sets and scale features.
        """
        # Perform train-test split
        self.X_train, self.X_test, self.y_train_df, self.y_test_df = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        logger.info(f"Splitting data with test size = {self.test_size} and random state = {self.random_state}.")

        # Feature scaling
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        # Ensure y_train and y_test are numpy arrays of integers
        self.y_train = self.y_train_df.astype(int).to_numpy()
        self.y_test = self.y_test_df.astype(int).to_numpy()

    def tune_model(self, model_name):
        """
        Tune hyperparameters for the model using GridSearchCV.
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} is not supported.")
            return None

        model = self.models[model_name]
        param_grid = self.hyperparameters[model_name]
        logger.info(f"Tuning {model_name} with GridSearchCV...")

        # Define scoring metrics with zero_division parameter
        scoring = {
            'f1_samples': make_scorer(f1_score, average='samples', zero_division=0)
        }

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.cv_folds,
            scoring=scoring,
            refit='f1_samples',
            return_train_score=True
        )

        grid_search.fit(self.X_train, self.y_train)

        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best F1 score (samples): {grid_search.best_score_}")

        # Log best parameters and score with MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric(f"{model_name}_best_f1_score", grid_search.best_score_)

        return grid_search.best_estimator_

    def train_model(self, model_name):
        """
        Train a model based on the model name using MultiOutputClassifier.
        """
        model = self.models.get(model_name)
        if not model:
            logger.error(f"Model {model_name} not found.")
            return None
        logger.info(f"Training {model_name} model.")
        model.fit(self.X_train, self.y_train)

        self.best_thresholds = self.find_best_threshold(model, self.X_test, self.y_test)

        # Log model with MLflow
        mlflow.sklearn.log_model(model, f"{model_name}_model")

        return model

    def find_best_threshold(self, model, X_val, y_val):
        """
        Find the best threshold that maximizes the F1 score on the validation set.
        Uses calibrated probabilities and vectorized operations for improved performance.
        """
        logger.info("Finding the best threshold for each label using the validation set.")

        y_val = y_val.astype(int)
        n_labels = y_val.shape[1]

        # Calibrate probabilities for all labels at once
        calibrated_models = [
            CalibratedClassifierCV(model.estimators_[i], method='sigmoid', cv='prefit')
            for i in range(n_labels)
        ]
        for i, cal_model in enumerate(calibrated_models):
            cal_model.fit(X_val, y_val[:, i])

        # Get calibrated probabilities for all labels
        calibrated_probas = np.array([
            cal_model.predict_proba(X_val)[:, 1] for cal_model in calibrated_models
        ]).T

        # Initialize arrays to store metrics
        thresholds = np.linspace(0, 1, 100)
        f1_scores = np.zeros((n_labels, len(thresholds)))

        # Vectorized computation of precision, recall, and F1 score
        for i, threshold in enumerate(thresholds):
            predictions = (calibrated_probas >= threshold).astype(int)
            true_positives = (predictions * y_val).sum(axis=0)
            predicted_positives = predictions.sum(axis=0)
            actual_positives = y_val.sum(axis=0)

            precision = np.divide(true_positives, predicted_positives,
                                  out=np.zeros_like(true_positives, dtype=float),
                                  where=predicted_positives != 0)
            recall = np.divide(true_positives, actual_positives,
                               out=np.zeros_like(true_positives, dtype=float),
                               where=actual_positives != 0)

            f1_scores[:, i] = np.divide(2 * precision * recall, precision + recall,
                                        out=np.zeros_like(precision, dtype=float),
                                        where=(precision + recall) != 0)

        # Find the best threshold for each label
        best_thresholds = thresholds[np.argmax(f1_scores, axis=1)]

        logger.info(f"Best thresholds for each label: {best_thresholds}")
        return best_thresholds

    def log_misclassifications(self, actual_labels, predicted_labels):
        """
        Logs misclassified samples by comparing actual and predicted labels.
        """
        for i in range(len(actual_labels)):
            if not np.array_equal(actual_labels[i], predicted_labels[i]):
                logger.info(f"Misclassified sample index {i}:")
                logger.info(f"Actual: {actual_labels[i]}")
                logger.info(f"Predicted: {predicted_labels[i]}")

    def evaluate_model_performance(self, model):
        logger.info(f"Evaluating {model.__class__.__name__} model...")

        y_pred = self.predict_with_threshold(model, self.X_test)

        # Ensure y_pred and y_test have the same shape
        if y_pred.shape != self.y_test.shape:
            logger.error(f"Shape mismatch: y_test has shape {self.y_test.shape}, but y_pred has shape {y_pred.shape}.")
            return None

        # Store overall metrics
        precision_list, recall_list, f1_list, accuracy_list, hamming_list = [], [], [], [], []
        class_metrics = {}

        for i in range(self.y_test.shape[1]):
            # Class-specific analysis
            precision = precision_score(self.y_test[:, i], y_pred[:, i], zero_division=0)
            recall = recall_score(self.y_test[:, i], y_pred[:, i], zero_division=0)
            f1 = f1_score(self.y_test[:, i], y_pred[:, i], zero_division=0)
            accuracy = accuracy_score(self.y_test[:, i], y_pred[:, i])
            hamming = hamming_loss(self.y_test[:, i], y_pred[:, i])

            # Store per-class metrics
            class_metrics[f'Label_{i}'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'hamming_loss': hamming
            }

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            accuracy_list.append(accuracy)
            hamming_list.append(hamming)

        # Calculate mean metrics
        mean_metrics = {
            'precision': np.mean(precision_list),
            'recall': np.mean(recall_list),
            'f1_score': np.mean(f1_list),
            'accuracy': np.mean(accuracy_list),
            'hamming_loss': np.mean(hamming_list),
            'best_thresholds': self.best_thresholds
        }

        # Log metrics with MLflow
        for metric_name, metric_value in mean_metrics.items():
            if metric_name != 'best_thresholds':
                mlflow.log_metric(f"mean_{metric_name}", metric_value)

        for label, metrics in class_metrics.items():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", metric_value)

        logger.info(f"Mean Precision: {mean_metrics['precision']:.4f}")
        logger.info(f"Mean Recall: {mean_metrics['recall']:.4f}")
        logger.info(f"Mean F1 Score: {mean_metrics['f1_score']:.4f}")
        logger.info(f"Mean Accuracy: {mean_metrics['accuracy']:.4f}")
        logger.info(f"Mean Hamming Loss: {mean_metrics['hamming_loss']:.4f}")

        # Log class-specific metrics
        for label, metrics in class_metrics.items():
            logger.info(f"{label} -> Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                        f"F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
                        f"Hamming Loss: {metrics['hamming_loss']:.4f}")

        # Log misclassifications
        self.log_misclassifications(self.y_test, y_pred)

        return mean_metrics, class_metrics

    def predict_with_threshold(self, model, input_data):
        """
        Predict with model and apply thresholds to classify outputs.
        """
        probas = model.predict_proba(input_data)

        # Use the best thresholds found during validation
        if self.best_thresholds is None:
            thresholds = [0.5] * len(probas)
        else:
            thresholds = self.best_thresholds

        predictions = np.zeros((input_data.shape[0], len(probas)))

        # Iterate over each label and apply the corresponding threshold
        for i, probs in enumerate(probas):
            # probs is (n_samples, n_classes) for each label
            if probs.shape[1] == 2:
                # Binary classification, take probability of class 1
                class1_probs = probs[:, 1]
            else:
                # Multiclass classification, adjust as needed
                class1_probs = probs.max(axis=1)
            predictions[:, i] = (class1_probs >= thresholds[i]).astype(int)

        return predictions

    def run_all_models(self):
        """
        Train and evaluate all models, with optional cross-validation and hyperparameter tuning.
        """
        self.train_test_split()

        results = {}
        for model_name in self.models:
            with mlflow.start_run(nested=True):
                mlflow.log_param("model_name", model_name)
                logger.info(f"Training and evaluating model: {model_name}")
                trained_model = self.tune_model(model_name)
                if trained_model:
                    mean_metrics, class_metrics = self.evaluate_model_performance(trained_model)
                    results[model_name] = mean_metrics

        logger.info("\nAll model results:\n")
        for model_name, metrics in results.items():
            logger.info(f"{model_name}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                        f"F1 Score: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
                        f"Hamming Loss: {metrics['hamming_loss']:.4f}")

        return results

    @gin.configurable
    def save_model(self, model, model_name, models_dir):
        # Ensure the directory exists
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, f'{model_name.lower()}_model.pkl')

        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Trained model saved at {model_path}")

        return model_path

    @gin.configurable
    def train_and_save_best_model(self, models_dir):
        try:
            mlflow.log_param("models_dir", models_dir)

            # Train and test the model
            self.train_test_split()

            # Train all models and get the results
            results = self.run_all_models()

            # Find the model with the best F1 score
            best_model_name = max(results, key=lambda x: results[x]['f1_score'])
            logger.info(f"Best model based on F1 Score: {best_model_name}")
            mlflow.log_param("best_model", best_model_name)

            # Train the best model
            trained_model = self.train_model(best_model_name)

            # Save the model (this was missing)
            model_path = self.save_model(trained_model, best_model_name, models_dir)
            mlflow.log_artifact(model_path)

            # Define input and output schema
            from mlflow.models import ModelSignature
            from mlflow.types.schema import Schema, ColSpec
            input_schema = Schema([ColSpec("double", f"feature_{i}") for i in range(self.X.shape[1])])
            output_schema = Schema([ColSpec("integer", f"label_{i}") for i in range(self.y.shape[1])])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            # Log and register the model
            mlflow.sklearn.log_model(
                sk_model=trained_model,
                artifact_path="model",
                registered_model_name="AmphibiansClassifier",
                signature=signature
            )
            logger.info(f"Model '{best_model_name}' registered as 'AmphibiansClassifier' in MLflow")

            # Save best thresholds
            thresholds_path = os.path.join(models_dir, f'{best_model_name.lower()}_thresholds.pkl')
            with open(thresholds_path, 'wb') as f:
                pickle.dump(self.best_thresholds, f)
            logger.info(f"Best thresholds saved at {thresholds_path}")
            mlflow.log_artifact(thresholds_path)

            return best_model_name, self
        except Exception as e:
            logger.error(f"Error during model training or saving: {e}")
            mlflow.log_param("error", str(e))
            raise
