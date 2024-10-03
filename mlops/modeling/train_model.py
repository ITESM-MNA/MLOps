import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, hamming_loss, precision_recall_curve

logger = logging.getLogger(__name__)

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

    def train_test_split(self):
        """
        Split the data into training and testing sets, and apply SMOTE to the training data.
        """
        # Perform train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        logger.info(f"Splitting data with test size = {self.test_size} and random state = {self.random_state}.")

        # Apply SMOTE individually to each label column
        logger.info("Applying SMOTE for oversampling each label column individually...")
        smote = SMOTE(random_state=self.random_state)

        if not isinstance(self.y_train, pd.DataFrame):
            self.y_train = pd.DataFrame(self.y_train)

        y_train_resampled = []
        X_resampled = None
        min_size = float('inf')

        for i in range(self.y_train.shape[1]):
            _, y_res = smote.fit_resample(self.X_train, self.y_train.iloc[:, i])
            min_size = min(min_size, len(y_res))

        for i in range(self.y_train.shape[1]):
            X_res, y_res = smote.fit_resample(self.X_train, self.y_train.iloc[:, i])
            y_train_resampled.append(y_res[:min_size])

            if X_resampled is None:
                X_resampled = X_res[:min_size]

        self.X_train = X_resampled
        self.y_train = np.column_stack(y_train_resampled)

        logger.info(f"After SMOTE: X_train shape = {self.X_train.shape}, y_train shape = {self.y_train.shape}")

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
        """
        logger.info("Finding the best threshold for each label using the validation set.")

        if isinstance(y_val, pd.DataFrame):
            y_val = y_val.to_numpy()

        y_val = y_val.astype(int)

        y_probs = [model.estimators_[i].predict_proba(X_val) for i in range(y_val.shape[1])]

        best_thresholds = []

        for i in range(y_val.shape[1]):
            probas = y_probs[i][:, 1]
            precisions, recalls, thresholds = precision_recall_curve(y_val[:, i], probas, pos_label=1)

            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

            best_idx = np.argmax(f1_scores)
            best_thresholds.append(thresholds[best_idx])

        logger.info(f"Best thresholds for each label: {best_thresholds}")
        return best_thresholds

    def evaluate_model_performance(self, model):
        """
        Evaluate model performance using several metrics.
        """
        logger.info(f"Evaluating {model.__class__.__name__} model...")

        y_pred = self.predict_with_threshold(model, self.X_test)

        if isinstance(self.y_test, pd.DataFrame):
            self.y_test = self.y_test.to_numpy()
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.to_numpy()

        if y_pred.shape != self.y_test.shape:
            logger.error(f"Shape mismatch: y_test has shape {self.y_test.shape}, but y_pred has shape {y_pred.shape}.")
            return None

        precision_list = []
        recall_list = []
        f1_list = []
        accuracy_list = []
        hamming_list = []

        self.y_test = self.y_test.astype(int)
        y_pred = y_pred.astype(int)

        for i in range(self.y_test.shape[1]):
            try:
                precision = precision_score(self.y_test[:, i], y_pred[:, i], labels=[0, 1], zero_division=0)
                recall = recall_score(self.y_test[:, i], y_pred[:, i], labels=[0, 1], zero_division=0)
                f1 = f1_score(self.y_test[:, i], y_pred[:, i], labels=[0, 1], zero_division=0)
                accuracy = accuracy_score(self.y_test[:, i], y_pred[:, i])
                hamming = hamming_loss(self.y_test[:, i], y_pred[:, i])
            except ValueError as e:
                logger.error(f"Error in calculating metrics for label {i}: {e}")
                continue

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            accuracy_list.append(accuracy)
            hamming_list.append(hamming)

        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)
        mean_f1 = np.mean(f1_list)
        mean_accuracy = np.mean(accuracy_list)
        mean_hamming = np.mean(hamming_list)

        logger.info(f"Mean Precision: {mean_precision:.4f}")
        logger.info(f"Mean Recall: {mean_recall:.4f}")
        logger.info(f"Mean F1 Score: {mean_f1:.4f}")
        logger.info(f"Mean Accuracy: {mean_accuracy:.4f}")
        logger.info(f"Mean Hamming Loss: {mean_hamming:.4f}")

        return mean_precision, mean_recall, mean_f1, mean_accuracy, mean_hamming

    def predict_with_threshold(self, model, X_test):
        """
        Predict using the loaded model and apply the best thresholds for each label.
        """
        logger.info("Applying best thresholds to predictions.")

        y_probs = [model.estimators_[i].predict_proba(X_test) for i in range(self.y_test.shape[1])]

        if len(y_probs) != self.y_test.shape[1]:
            logger.error(f"Number of predicted labels ({len(y_probs)}) does not match the number of actual labels ({self.y_test.shape[1]}).")
            return None

        predictions = []

        for i, prob in enumerate(y_probs):
            prob_class_1 = np.array([p[1] for p in prob])
            pred = (prob_class_1 >= self.best_thresholds[i]).astype(int)
            predictions.append(pred)

        predictions = np.column_stack(predictions)

        logger.info(f"Prediction shape: {predictions.shape}")
        return predictions

    def run_all_models(self):
        """
        Train and evaluate all models, with optional cross-validation and hyperparameter tuning.
        """
        self.train_test_split()

        results = {}
        for model_name in self.models:
            logger.info(f"Training and evaluating model: {model_name}")
            trained_model = self.train_model(model_name)
            if trained_model:
                precision, recall, f1, accuracy, hamming = self.evaluate_model_performance(trained_model)
                results[model_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'accuracy': accuracy,
                    'hamming_loss': hamming
                }

        logger.info("\nAll model results:\n")
        for model_name, metrics in results.items():
            logger.info(f"{model_name}: {metrics}")

        return results
