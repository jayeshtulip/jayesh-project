from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

# Define input schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
model = joblib.load(model_path)

# Species mapping
species_map = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "ML Prediction API is live!"}

# Prediction endpoint
@app.post("/predict")
def predict_species(features: IrisFeatures):
    input_data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]

    prediction = model.predict(input_data)[0]
    return {
        "predicted_class": int(prediction),
        "predicted_species": species_map.get(int(prediction), "unknown")
    }

    


