from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
import logging

# Initialize logging
logger = logging.getLogger(__name__)


class TrainModel:
    """
    Class for training and testing different classification models, with optional hyperparameter tuning.
    Supports multi-label classification using OneVsRestClassifier and MultiLabelBinarizer.
    """

    def __init__(self, X, y, cv_enabled=False, cv_folds=5, tune_hyperparameters=False):
        self.X = X
        self.y = y
        self.cv_enabled = cv_enabled
        self.cv_folds = cv_folds
        self.tune_hyperparameters = tune_hyperparameters
        self.mlb = MultiLabelBinarizer()
        self.y_binarized = self.mlb.fit_transform(self.y.values)

        self.models = {
            'Logistic Regression': OneVsRestClassifier(LogisticRegression()),
            'Random Forest': OneVsRestClassifier(RandomForestClassifier()),
            'Decision Tree': OneVsRestClassifier(DecisionTreeClassifier()),
            'SVM': OneVsRestClassifier(SVC()),
        }

        self.hyperparameters = {
            'Logistic Regression': {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']},
            'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]},
            'Decision Tree': {'max_depth': [5, 10, 20, None], 'min_samples_split': [2, 5, 10]},
            'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        }

    def train_test_split(self, test_size=0.3, random_state=42):
        """
        Split the dataset into training and testing sets.
        """
        logger.info(f"Splitting data with test size = {test_size} and random state = {random_state}.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y_binarized,
                                                                                test_size=test_size,
                                                                                random_state=random_state)

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

        grid_search = GridSearchCV(model, param_grid, cv=self.cv_folds, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def cross_validate_model(self, model_name):
        """
        Perform cross-validation for a specific model.
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} is not supported.")
            return None

        model = self.models[model_name]
        logger.info(f"Performing {self.cv_folds}-fold cross-validation for {model_name}.")
        scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv_folds, scoring='accuracy')
        mean_score = np.mean(scores)
        logger.info(f"Cross-validation accuracy for {model_name}: {mean_score:.4f}")
        return mean_score

    def train_model(self, model_name):
        """
        Train a specific model by its name.
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} is not supported.")
            return None

        model = self.models[model_name]
        logger.info(f"Training {model_name} model.")
        model.fit(self.X_train, self.y_train)
        return model

    def evaluate_model(self, model, model_name):
        """
        Evaluate the trained model on the test set.
        """
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        logger.info(f"\nEvaluation for {model_name}:")
        logger.info(f"Accuracy: {accuracy:.4f}")

        # Log detailed classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(self.y_test, y_pred, target_names=self.mlb.classes_))

        # Log confusion matrix
        logger.info("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(self.y_test.argmax(axis=1), y_pred.argmax(axis=1))
        logger.info(f"\n{conf_matrix}\n")

        return accuracy

    def run_all_models(self):
        """
        Train and evaluate all models, with optional cross-validation and hyperparameter tuning.
        """
        results = {}
        self.train_test_split()  # Split the data before running models

        for model_name in self.models:
            if self.tune_hyperparameters:
                logger.info(f"Tuning hyperparameters for {model_name}")
                best_model = self.tune_model(model_name)
                accuracy = self.evaluate_model(best_model, model_name)
                results[f"{model_name} (Tuned)"] = accuracy
            else:
                logger.info(f"Training and evaluating model: {model_name}")
                trained_model = self.train_model(model_name)
                if trained_model:
                    accuracy = self.evaluate_model(trained_model, model_name)
                    results[model_name] = accuracy

        logger.info("\nAll model results:\n")
        for model_name, accuracy in results.items():
            logger.info(f"{model_name}: {accuracy:.4f}")

        return results
