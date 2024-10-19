import argparse
import logging
import os
import gin
import mlflow
from datetime import datetime
# from sklearnex import patch_sklearn

from feature_engineering.data_preprocessing import DataPreprocessing
from modeling.prediction import Predictor
from modeling.train_model import TrainModel


# Function to configure logging to both console and file
@gin.configurable
def configure_logging(reports_dir):

    # Ensure the reports directory exists
    os.makedirs(reports_dir, exist_ok=True)

    # Path to the log file inside reports_dir
    log_file_path = os.path.join(reports_dir, 'pipeline.log')

    # Set up logging with both console and file handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(),  # Logs to console
            logging.FileHandler(log_file_path, mode='w')  # Logs to file
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Logs will be saved to {log_file_path}")
    return logger


@gin.configurable
def configure_mlflow(experiment_name, tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logging.info(f"MLflow configured with experiment '{experiment_name}' and tracking URI '{tracking_uri}'")


def ensure_directories_exist():
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)


def generate_run_name(model_name, hyperparameters):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    hyperparam_str = "_".join([f"{k[:3]}{v}" for k, v in hyperparameters.items()][:3])  # Limit to first 3 hyperparameters
    return f"{model_name}_{hyperparam_str}_{timestamp}"


@gin.configurable
def run(config_path: str):
    try:
        # Load and preprocess the data
        X_pca, y = DataPreprocessing.load_and_preprocess_data()

        # Initialize TrainModel
        train_model = TrainModel(X_pca, y, config_path)

        # Train and evaluate all models
        results = train_model.run_all_models()

        # Find the best model
        best_model_name = max(results, key=lambda x: results[x]['f1_score'])
        logger.info(f"Best model based on F1 Score: {best_model_name}")

        # Get the best hyperparameters
        best_hyperparameters = train_model.models[best_model_name].get_params()

        # Generate a meaningful run name
        run_name = generate_run_name(best_model_name, best_hyperparameters)

        # Start the MLflow run with the generated name
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"Started MLflow run '{run_name}' with ID: {run.info.run_id}")

            # Log the best model name and hyperparameters
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_params(best_hyperparameters)

            # Train the best model
            trained_model = train_model.train_model(best_model_name)

            # Save the model
            model_path = train_model.save_model(trained_model, best_model_name, 'models')
            mlflow.log_artifact(model_path)

            # Load the trained model
            predictor = Predictor(model_path=model_path)
            predictor.load_model()

            # Evaluate model performance
            mean_metrics, class_metrics = train_model.evaluate_model_performance(predictor.model)

            # Log model evaluation metrics
            logger.info(f"Evaluation metrics: {mean_metrics}")

            # Log metrics to MLflow
            for key, value in mean_metrics.items():
                if key != 'best_thresholds':
                    mlflow.log_metric(key, value)

            # Run predictions on a few samples
            sample_indices = [0, 1, 5, 10]  # You can adjust these indices as needed
            Predictor.run_predictions(predictor, train_model, sample_indices)

    except Exception as e:
        logger.error(f"An error occurred in the pipeline: {e}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the Gin config file')
    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories_exist()

    # Load Gin configuration
    try:
        gin.parse_config_file(args.config)
        logger = configure_logging()
        configure_mlflow()
        logger.info("Gin configuration and MLflow tracking configured successfully.")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        raise

    # Execute the main run function
    run()


