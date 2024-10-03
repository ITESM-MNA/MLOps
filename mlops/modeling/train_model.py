import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score, hamming_loss,
                             precision_recall_curve, make_scorer)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TrainModel:
    def __init__(self, X, y, test_size=0.3, random_state=42):
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_size = test_size
        self.random_state = random_state
        self.best_thresholds = None
        self.models = {
            "Logistic Regression": MultiOutputClassifier(LogisticRegression()),
            "Random Forest": MultiOutputClassifier(RandomForestClassifier()),
            "Decision Tree": MultiOutputClassifier(DecisionTreeClassifier()),
            "SVM": MultiOutputClassifier(SVC(probability=True))
        }

        self.hyperparameters = {
            'Logistic Regression': {
                'estimator__C': [0.1, 1, 10],
                'estimator__solver': ['liblinear', 'saga'],
                'estimator__class_weight': ['balanced']
            },
            'Random Forest': {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': [5, 10, None],
                'estimator__min_samples_split': [2, 5],
                'estimator__min_samples_leaf': [1, 2],
                'estimator__class_weight': ['balanced', 'balanced_subsample']
            },
            'Decision Tree': {
                'estimator__max_depth': [5, 10, None],
                'estimator__min_samples_split': [2, 5],
                'estimator__min_samples_leaf': [1, 2],
                'estimator__class_weight': ['balanced']
            },
            'SVM': {
                'estimator__C': [0.1, 1, 10],
                'estimator__kernel': ['linear', 'rbf'],
                'estimator__gamma': ['scale', 'auto'],
                'estimator__class_weight': ['balanced']
            }
        }

    def train_test_split(self):
        """
        Split the data into training and testing sets.
        """
        # Perform train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        logger.info(f"Splitting data with test size = {self.test_size} and random state = {self.random_state}.")

    def tune_model(self, model_name, cv_folds=5):
        """
        Tune hyperparameters for the model using GridSearchCV.
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} is not supported.")
            return None

        model = self.models[model_name]
        param_grid = self.hyperparameters[model_name]
        logger.info(f"Tuning {model_name} with GridSearchCV...")

        # Define scoring metrics
        scoring = {
            'f1_macro': make_scorer(f1_score, average='macro'),
            'hamming_loss': make_scorer(hamming_loss)
        }

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            refit='f1_macro',  # Refitting based on the F1 macro score
            return_train_score=True  # Optionally return training scores for analysis
        )

        grid_search.fit(self.X_train, self.y_train)

        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best F1 score (macro): {grid_search.best_score_}")
        if 'hamming_loss' in grid_search.cv_results_:
            logger.info(
                f"Best Hamming Loss: {grid_search.cv_results_['mean_test_hamming_loss'][grid_search.best_index_]}"
            )

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

        return model

    def find_best_threshold(self, model, X_val, y_val):
        """
        Find the best threshold that maximizes the F1 score on the validation set.
        Also calibrate the probabilities using Platt Scaling.
        """
        logger.info("Finding the best threshold for each label using the validation set.")

        if isinstance(y_val, pd.DataFrame):
            y_val = y_val.to_numpy()

        y_val = y_val.astype(int)

        y_probs = [model.estimators_[i].predict_proba(X_val) for i in range(y_val.shape[1])]
        best_thresholds = []

        for i in range(y_val.shape[1]):
            probas = y_probs[i][:, 1]

            # Apply Platt Scaling for probability calibration
            calibrated_model = CalibratedClassifierCV(model.estimators_[i], method='sigmoid', cv='prefit')
            calibrated_model.fit(X_val, y_val[:, i])
            calibrated_probas = calibrated_model.predict_proba(X_val)[:, 1]

            precisions, recalls, thresholds = precision_recall_curve(y_val[:, i], calibrated_probas, pos_label=1)

            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

            best_idx = np.argmax(f1_scores)
            if best_idx < len(thresholds):
                best_thresholds.append(thresholds[best_idx])
            else:
                best_thresholds.append(0.5)

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

        if isinstance(self.y_test, pd.DataFrame):
            self.y_test = self.y_test.to_numpy()

        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.to_numpy()

        # Convert y_test to integers (since y_pred is numeric)
        self.y_test = self.y_test.astype(int)

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
            # probs is (n_samples, 2) for each label, we take probs[:, 1] to get the probabilities for class 1
            predictions[:, i] = (probs[:, 1] >= thresholds[i]).astype(int)

        return predictions

    def run_all_models(self):
        """
        Train and evaluate all models, with optional cross-validation and hyperparameter tuning.
        """
        self.train_test_split()

        results = {}
        for model_name in self.models:
            logger.info(f"Training and evaluating model: {model_name}")
            trained_model = self.tune_model(model_name)
            if trained_model:
                mean_metrics, class_metrics = self.evaluate_model_performance(trained_model)  # Capture both returns
                results[model_name] = mean_metrics  # Store only mean metrics

        logger.info("\nAll model results:\n")
        for model_name, metrics in results.items():
            logger.info(f"{model_name}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                        f"F1 Score: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
                        f"Hamming Loss: {metrics['hamming_loss']:.4f}")

        return results
