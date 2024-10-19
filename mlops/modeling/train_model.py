import logging
import numpy as np
import os
import pickle
import gin

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
    def __init__(self, X, y, test_size=0.3, random_state=42):
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_df = None
        self.y_test_df = None
        self.test_size = test_size
        self.random_state = random_state
        self.best_thresholds = None
        self.models = {
            "Logistic Regression": MultiOutputClassifier(LogisticRegression()),
            "Random Forest": MultiOutputClassifier(RandomForestClassifier()),
            "Extra Trees": MultiOutputClassifier(ExtraTreesClassifier()),
            "AdaBoost": MultiOutputClassifier(AdaBoostClassifier()),
            "Gradient Boosting": MultiOutputClassifier(GradientBoostingClassifier()),
            "Decision Tree": MultiOutputClassifier(DecisionTreeClassifier()),
            "SVM": MultiOutputClassifier(SVC(probability=True))
        }

        self.hyperparameters = {
            'Logistic Regression': {
                'estimator__C': [0.1, 1, 10],
                'estimator__solver': ['liblinear', 'saga'],
                'estimator__class_weight': ['balanced'],
                'estimator__max_iter': [100, 200, 500, 1000]
            },
            'Random Forest': {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': [5, 10, None],
                'estimator__min_samples_split': [2, 5],
                'estimator__min_samples_leaf': [1, 2],
                'estimator__class_weight': ['balanced', 'balanced_subsample']
            },
            'Extra Trees': {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': [5, 10, None],
                'estimator__min_samples_split': [2, 5],
                'estimator__min_samples_leaf': [1, 2],
                'estimator__class_weight': ['balanced', 'balanced_subsample']
            },
            'AdaBoost': {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__learning_rate': [0.01, 0.1, 1],
                'estimator__algorithm': ['SAMME', 'SAMME.R']
            },
            'Gradient Boosting': {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__learning_rate': [0.01, 0.1, 1],
                'estimator__max_depth': [3, 5, 7],
                'estimator__min_samples_split': [2, 5],
                'estimator__min_samples_leaf': [1, 2]
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

        # Define scoring metrics with zero_division parameter
        scoring = {
            'f1_samples': make_scorer(f1_score, average='samples', zero_division=0)
        }

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            refit='f1_samples',
            return_train_score=True
        )

        grid_search.fit(self.X_train, self.y_train)

        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best F1 score (samples): {grid_search.best_score_}")

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
            # Train and test the model
            self.train_test_split()

            # Train all models and get the results
            results = self.run_all_models()

            # Find the model with the best F1 score
            best_model_name = max(results, key=lambda x: results[x]['f1_score'])
            logger.info(f"Best model based on F1 Score: {best_model_name}")

            # Train the best model and save it
            trained_model = self.train_model(best_model_name)
            self.save_model(trained_model, best_model_name)

            # Save best thresholds
            with open(os.path.join(models_dir, f'{best_model_name.lower()}_thresholds.pkl'), 'wb') as f:
                pickle.dump(self.best_thresholds, f)
            logger.info(f"Best thresholds saved at {os.path.join(models_dir, f'{best_model_name.lower()}_thresholds.pkl')}")

            return best_model_name, self
        except Exception as e:
            logger.error(f"Error during model training or saving: {e}")
            raise
