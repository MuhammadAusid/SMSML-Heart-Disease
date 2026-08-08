
from fastapi import FastAPI
from pydantic import BaseModel

from inference import predict_heart_disease


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Real inference API menggunakan tuned Random Forest model",
    version="1.0.0"
)


class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API is running"
    }


@app.post("/predict")
def predict(patient: PatientData):

    result = predict_heart_disease(
        patient.model_dump()
    )

    return result
