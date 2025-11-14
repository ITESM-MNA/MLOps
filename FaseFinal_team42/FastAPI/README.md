# FastAPI Service for Model Serving

## Overview
This FastAPI service exposes a machine learning model via an HTTP API. The service includes an endpoint for making predictions and validates input data using Pydantic.

## Endpoints

### POST `/predict`
- **Description**: Predicts the output based on input features.
- **Request Body**:
  ```json
  {
      "features": [float, float, ...]
  }
  ```
- **Response**:
  ```json
  {
      "prediction": int,
      "probability": float
  }
  ```
- **Error Handling**:
  - Returns `400` if the input is invalid.
  - Returns `500` for internal server errors.

## Model Artifact
- **Path**: `models:/insurance_model/v1`
- **Version**: `v1`

## Running the Service
1. Install dependencies:
   ```bash
   pip install fastapi uvicorn scikit-learn joblib
   ```
2. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
3. Open the Swagger documentation at:
   [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Testing with Postman
1. Import the Swagger schema from:
   [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)
2. Use the `/predict` endpoint to test predictions.

## Notes
- Ensure the model artifact is available at the specified path before starting the service.
- Update the `MODEL_PATH` in `main.py` if the model location changes.
