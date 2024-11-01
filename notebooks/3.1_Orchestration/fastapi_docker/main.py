from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import numpy as np
import uvicorn
from sklearn.datasets import load_wine

# Load the saved model
with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load target names for response
data = load_wine()
target_names = data.target_names

# Define the input data format for prediction
class WineData(BaseModel):
    features: List[float]

# Initialize FastAPI app
app = FastAPI()

# Define prediction endpoint
@app.post("/predict")
def predict(wine_data: WineData):
    # Validate input length
    if len(wine_data.features) != model.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Input must contain {model.n_features_in_} features."
        )

    # Make prediction
    prediction = model.predict([wine_data.features])[0]
    prediction_name = target_names[prediction]
    
    return {"prediction": int(prediction), "prediction_name": prediction_name}

# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Wine classification model API"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
