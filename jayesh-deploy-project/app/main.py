from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# ✅ Load model
import os

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "model", "model.pkl")

model = joblib.load(model_path)

# ✅ Define app
app = FastAPI()

# ✅ Input schema
class IrisFeatures(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# ✅ Root route
@app.get("/")
def read_root():
    return {"message": "ML Prediction API is live!"}

# ✅ Prediction route
@app.post("/predict")
def predict(features: IrisFeatures):
    X = np.array([[features.feature1, features.feature2, features.feature3, features.feature4]])
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}
