# FastAPI Machine Learning API - Wine Classification

This repository demonstrates using FastAPI to serve a machine learning model for classifying wine types based on chemical properties. The model is trained on the Wine dataset from `scikit-learn`, using a `LogisticRegression` classifier.

## Overview

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints. FastAPI is ideal for machine learning APIs because it’s quick to set up, easy to test, and provides high performance.

In this example, we:
- Train a Logistic Regression model to classify wine types using the Wine dataset.
- Serve the model through a FastAPI application.
- Dockerize the application to make it portable and easy to deploy.

## Requirements

- Python 3.7+
- FastAPI
- scikit-learn
- Uvicorn (for serving the FastAPI app)
- Docker (for containerization)

## Setup

### 1. Install Dependencies

Install FastAPI, scikit-learn, and Uvicorn.

```bash
pip install fastapi scikit-learn uvicorn
```

### 2. Train the Model

The `train_model.py` script loads the Wine dataset, trains a `LogisticRegression` model, and saves it to a file called `wine_model.pkl`.

```python
# train_model.py
import pickle
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate and save the model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Logistic Regression model trained with an accuracy of: {accuracy:.2f}")

# Save the model
with open("wine_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as 'wine_model.pkl'")
```

Run the script to create and save the model:

```bash
python train_model.py
```

### 3. Create the FastAPI Application

The `main.py` file loads the saved model and uses FastAPI to serve predictions.

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
from sklearn.datasets import load_wine
import numpy as np

# Load the model
with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the target names (class labels)
data = load_wine()
target_names = data.target_names

# Define the input data format for prediction
class WineData(BaseModel):
    features: List[float]

# Initialize FastAPI
app = FastAPI()

# Prediction endpoint
@app.post("/predict")
def predict(wine_data: WineData):
    if len(wine_data.features) != model.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Input must contain {model.n_features_in_} features."
        )
    # Predict
    prediction = model.predict([wine_data.features])[0]
    prediction_name = target_names[prediction]
    return {"prediction": int(prediction), "prediction_name": prediction_name}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Wine classification model API"}
```

### 4. Run the Application

To run the application, use Uvicorn, an ASGI server for Python web applications.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Testing the API

You can test the API by sending a POST request to `http://localhost:8000/predict` with a JSON body containing wine features.

#### Example Request in PowerShell (or you can use Postman)

```powershell
$headers = @{
    "Content-Type" = "application/json"
}

$body = @{
    "features" = @(13.2, 2.7, 2.36, 21, 100, 2.98, 3.15, 0.22, 2.26, 6.5, 1.05, 3.33, 820)
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Headers $headers -Body $body
```

### 6. Dockerize the Application

The `Dockerfile` makes it easy to build and run the application in a container.

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files to container
COPY wine_model.pkl /app/
COPY main.py /app/

# Install dependencies
RUN pip install fastapi uvicorn scikit-learn pydantic

# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run the Docker container:

```bash
# Build the Docker image
docker build -t wine-classification-api .

# Run the Docker container
docker run -p 8000:8000 wine-classification-api
```

### API Endpoints

- **GET `/`**: Basic endpoint to check if the server is running.
- **POST `/predict`**: Predicts the class of the wine given a list of features.

#### Example Response

A successful prediction request will return a JSON response like this:

```json
{
  "prediction": 1,
  "prediction_name": "class_1"
}
```

## FastAPI Features Used

- **Data Validation**: FastAPI uses Pydantic to validate incoming request data, ensuring feature list length is correct.
- **Asynchronous Processing**: FastAPI is built for async operation, making it highly performant for production.
