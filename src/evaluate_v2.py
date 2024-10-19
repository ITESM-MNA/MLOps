import pandas as pd
import sys
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import os

def evaluate_model(model_path, X_test_path, y_test_path, output_path):
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, zero_division=1)
    cm = confusion_matrix(y_test, predictions)
    write_evaluation_report(output_path, report, cm)

def write_evaluation_report(file_path, report, confusion_matrix):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix))

if __name__ == '__main__':
    model_path = sys.argv[1]
    X_test_path = sys.argv[2]
    y_test_path = sys.argv[3]
    output_path = sys.argv[4]
    evaluate_model(model_path, X_test_path, y_test_path, output_path)
