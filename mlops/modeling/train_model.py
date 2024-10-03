from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np

# Initialize logging
logger = logging.getLogger(__name__)


class TrainModel:
    """
    Class for training and testing different classification models, with optional cross-validation.
    Supports multi-label classification using OneVsRestClassifier and MultiLabelBinarizer.
    """

    def __init__(self, X, y, cv_enabled=False, cv_folds=5):
        """
        Initialize the TrainModel class with feature matrix X, labels y,
        and optional cross-validation settings.

        :param cv_enabled: Whether to use cross-validation (default: False)
        :param cv_folds: Number of folds for cross-validation (default: 5)
        """
        self.X = X
        self.y = y
        self.cv_enabled = cv_enabled
        self.cv_folds = cv_folds
        self.mlb = MultiLabelBinarizer()
        self.y_binarized = self.mlb.fit_transform(self.y.values)

        self.models = {
            'Logistic Regression': OneVsRestClassifier(LogisticRegression()),
            'Random Forest': OneVsRestClassifier(RandomForestClassifier()),
            'Decision Tree': OneVsRestClassifier(DecisionTreeClassifier()),
            'SVM': OneVsRestClassifier(SVC()),
        }

    def train_test_split(self, test_size=0.3, random_state=42):
        """
        Split the dataset into training and testing sets.
        """
        logger.info(f"Splitting data with test size = {test_size} and random state = {random_state}.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y_binarized,
                                                                                test_size=test_size,
                                                                                random_state=random_state)

    def cross_validate_model(self, model_name):
        """
        Perform cross-validation for a specific model.
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} is not supported.")
            return None

        model = self.models[model_name]
        logger.info(f"Performing {self.cv_folds}-fold cross-validation for {model_name}.")
        scores = cross_val_score(model, self.X, self.y_binarized, cv=self.cv_folds, scoring='accuracy')
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

    def evaluate_model(self, model):
        """
        Evaluate the trained model on the test set.
        """
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        logger.info(f"Accuracy of the model: {accuracy}")
        logger.info("Classification report:\n" + classification_report(self.y_test, y_pred))
        logger.info(f"Confusion matrix:\n{confusion_matrix(self.y_test.argmax(axis=1), y_pred.argmax(axis=1))}")
        return accuracy

    def run_all_models(self):
        """
        Train and evaluate all models, with optional cross-validation, and compare their performance.
        """
        results = {}
        self.train_test_split()

        for model_name in self.models:
            if self.cv_enabled:
                # Perform cross-validation
                logger.info(f"Running cross-validation for {model_name}")
                cv_score = self.cross_validate_model(model_name)
                results[f"{model_name} (CV)"] = cv_score
            else:
                # Train and evaluate without cross-validation
                logger.info(f"Training and evaluating model: {model_name}")
                trained_model = self.train_model(model_name)
                if trained_model:
                    accuracy = self.evaluate_model(trained_model)
                    results[model_name] = accuracy

        logger.info("All model results: " + str(results))
        return results
