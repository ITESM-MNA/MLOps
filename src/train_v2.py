import yaml
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow

def load_params():
    with open("params.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

params = load_params()

def train_model(X_train_path, y_train_path, model_type):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    model_params = params['models'][model_type]

    if model_type == 'logistic_regression':
        model = LogisticRegression(**model_params)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**model_params)

    mlflow.set_experiment(params['mlflow']['experiment_name'])
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])

    with mlflow.start_run():
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, f"{model_type}_model")

    return model

if __name__ == '__main__':
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_type = sys.argv[3]
    model_dir = params['data']['models']
    model_path = f"{model_dir}/{model_type}_model.pkl"

    model = train_model(X_train_path, y_train_path, model_type)
    joblib.dump(model, model_path)
