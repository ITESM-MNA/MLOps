from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
import os

app = FastAPI(
    title="Team42 ML Model API",
    description="API para realizar predicciones usando el modelo final de Gradient Boosting",
    version="1.0.0"
)

MODEL_PATH = "FaseFinal_team42/models/final_model_GradBoost.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

#Mismo orden que en el modelo
class ModelInput(BaseModel):
    ContrCarPol: float
    NumCarPol: float
    ContrFirePol: float
    DemAvgIncome: float
    DemMidInc: float
    DemLoLeEdu: float
    DemHiLeEdu: float
    DemLowestInc: float
    ContrPrivIns: float
    CMainType: float
    CAR_CROSS: float
    FIRE_CROSS: float  

@app.get("/")
def root():
    return {"message": "API funcionando. Ir a /docs para probar /predict"}

@app.post("/predict")
def predict(input_data: ModelInput):
    try:
        # Pydantic -> dict -> DataFrame con las columnas en el orden definido arriba
        df = pd.DataFrame([input_data.dict()])

        # Predicción
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]

        return {
            "prediction": int(pred),
            "probability": float(proba)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
