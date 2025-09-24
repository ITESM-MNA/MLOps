
# Modeling Pipeline

This guide provides instructions on setting up and running the Modeling Pipeline, which includes training, logging, tuning, evaluation, and serving a machine learning model using MLFlow and MinIO. Follow these steps to get the services running and perform key activities in the modeling pipeline.

## Prerequisites

Make sure you have the following software installed:
- Docker
- Docker Compose

Ensure you have access to the necessary environment variables (e.g., MINIO_ACCESS_KEY) and a configured `config.env` file.

## Setting Up and Running the Services

1. **Start the services with Docker Compose:**

```bash
docker-compose --env-file config.env up -d --build
```

This command will build and start the required services, including MLFlow and MinIO, in detached mode.

2. **Access MLFlow at:**

```bash
http://localhost:9001
```

MLFlow’s UI can be accessed through this address, where you can track experiments, view logs, and manage model versions.

3. **Configure MinIO Access Key:**

Retrieve the MinIO Access Key and store it in the `config.env` file under the variable `MINIO_ACCESS_KEY`.

```bash
export MINIO_ACCESS_KEY=your_minio_access_key
```

Make sure to save this key correctly to ensure proper access to MinIO storage.

4. **Stop the services:**

To stop the services when you are done:

```bash
docker-compose down
```

5. **Restart the services if necessary:**

```bash
docker-compose --env-file config.env up -d --build
```

This will bring the services back up using the latest configuration.

6. **If you need to specify a custom Docker Compose file:**

```bash
docker-compose -f docker-compose.yml --env-file config.env up -d --build
```

## 1.3. Modeling Pipeline Steps

### 1.3.1. Training

Once the services are up and running, you can initiate model training. This step will involve training the machine learning model and logging key metrics, parameters, and artifacts using MLFlow. Ensure that the training code is properly configured to log outputs to MLFlow.

### 1.3.2. Logging

MLFlow will automatically log the following:
- Hyperparameters
- Metrics such as accuracy, loss, etc.
- Model artifacts (such as model weights)
  
You can view the logs in MLFlow’s UI on `localhost:9001`.

### 1.3.3. Tuning

To tune the model, you can experiment with different hyperparameters and log the results to MLFlow. This will allow you to compare runs and choose the best configuration for your model.

### 1.3.4. Evaluation

After training and tuning, the model can be evaluated based on predefined metrics. MLFlow will track evaluation metrics, and you can compare them across different models and versions.

### 1.3.5. Serving

Finally, you can serve the trained model using MLFlow’s model serving capabilities. Ensure that the necessary services are running and that your model is available for inference.

---

## Example: Training a Logistic Regression Model with the Wine Dataset

In this section, we will demonstrate how to train a simple Logistic Regression model using the Wine dataset, log the results with MLFlow, and track the process.

### Prerequisites

Ensure that you have the Wine dataset (`wine_quality_df.csv`) available in your project directory.

### Step-by-Step Guide

1. **Prepare the Python Script for Training:**

Create a Python script (e.g., `train_logistic_regression.py`) with the following content:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Load the Wine dataset
data = pd.read_csv('wine_quality_df.csv')

# Prepare data
X = data.drop(columns=['quality'])  # Features
y = data['quality']  # Target variable

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up MLFlow experiment
mlflow.set_experiment("Wine_Quality_Experiment")

with mlflow.start_run():
    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log parameters and metrics to MLFlow
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("max_iter", 100)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Model accuracy: {accuracy}")

```

This script performs the following tasks:
- Loads the Wine dataset.
- Splits it into training and testing sets.
- Trains a Logistic Regression model.
- Logs key parameters and metrics (like accuracy) to MLFlow.
- Logs the trained model as an artifact in MLFlow.

2. **Run the Script:**

Once the services are running, execute the script from the command line:

```bash
python train_logistic_regression.py
```

This will train the model, log the results, and store the model in MLFlow.

3. **View Results in MLFlow:**

Navigate to `http://localhost:9001` to view the logged metrics, parameters, and model artifacts in MLFlow.

4. **Tuning the Model:**

To tune the model, modify the hyperparameters in the script (e.g., change the `max_iter` value), rerun the script, and compare the results in MLFlow.

5. **Serving the Model:**

After evaluating the model, you can serve it using MLFlow’s model serving capabilities. Use the following command to deploy the model:

```bash
mlflow models serve -m "runs:/<RUN_ID>/model" -p 5001
```

Replace `<RUN_ID>` with the ID of the run corresponding to the best model version in MLFlow.

6. **Access the Served Model:**

Once the model is being served, you can send inference requests to the model at `http://localhost:5001/invocations`.
