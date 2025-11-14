from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import os

# Initialize FastAPI app
app = FastAPI()

# Define input schema using Pydantic
class PredictionInput(BaseModel):
    features: List[float]

# Define output schema using Pydantic
class PredictionOutput(BaseModel):
    prediction: int
    probability: float

# Load the model artifact
MODEL_PATH = "C:\\Users\\mizlop\\OneDrive - SAS\\Documents\\SAS_git\\MLOps\\FaseFinal_team42\\models\\final_model_GradBoost.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """Endpoint to make predictions using the loaded model."""
    try:
        # Ensure the input features are valid
        if len(input_data.features) != model.n_features_in_:
            raise HTTPException(status_code=400, detail=f"Expected {model.n_features_in_} features, got {len(input_data.features)}")

        # Make prediction
        prediction = model.predict([input_data.features])[0]
        probability = model.predict_proba([input_data.features])[0].max()

        return PredictionOutput(prediction=int(prediction), probability=float(probability))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
