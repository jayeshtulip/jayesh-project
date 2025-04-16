from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("app/model/model.pkl")  # path relative to root

app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.get("/")
def read_root():
    return {"message": "ML Prediction API is live!"}

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.feature1, data.feature2, data.feature3]])
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
