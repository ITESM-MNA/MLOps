import joblib

MODEL_PATH = "../models/final_model_GradBoost.joblib"

model = joblib.load(MODEL_PATH)

print(" Número de features:", model.n_features_in_)

try:
    print("Nombres de features:")
    for f in model.feature_names_in_:
        print(f)
except AttributeError:
    print("El modelo NO tiene atributo feature_names_in_.")
