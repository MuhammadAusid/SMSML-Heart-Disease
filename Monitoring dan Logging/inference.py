
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Heart Disease Prediction API",
    description="API untuk prediksi penyakit jantung menggunakan Random Forest",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "Membangun_model",
        "saved_model",
        "heart_model.pkl"
    )
)

print("Model Path :", MODEL_PATH)
print("Model Exists :", os.path.exists(MODEL_PATH))

model = joblib.load(MODEL_PATH)


class HeartData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running"
    }


@app.post("/predict")
def predict(data: HeartData):

    df = pd.DataFrame([data.dict()])

    prediction = int(model.predict(df)[0])

    return {
        "prediction": prediction
    }
