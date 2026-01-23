from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Wine Quality Inference API")

# Load trained model
model = joblib.load("model.pkl")

# Input schema
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# Prediction endpoint
@app.post("/predict")
def predict_wine_quality(features: WineFeatures):
    
    # Convert input to model format
    input_data = np.array([[
        features.fixed_acidity,
        features.volatile_acidity,
        features.citric_acid,
        features.residual_sugar,
        features.chlorides,
        features.free_sulfur_dioxide,
        features.total_sulfur_dioxide,
        features.density,
        features.pH,
        features.sulphates,
        features.alcohol
    ]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Return response in required format
    return {
        "name": "Nandana",
        "roll_no": "2022BCS0005",
        "wine_quality": int(round(prediction))
    }
