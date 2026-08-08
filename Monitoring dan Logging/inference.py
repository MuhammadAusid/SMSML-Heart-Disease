from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import time

from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

app = FastAPI(
    title="Heart Disease Prediction API",
    description="API untuk prediksi penyakit jantung menggunakan Random Forest",
    version="1.0.0"
)

REQUEST_COUNTER = Counter('model_predictions_total', 'Total prediksi yang dilakukan')

LATENCY_HISTOGRAM = Histogram('model_prediction_latency_seconds', 'Waktu latensi prediksi')

POSITIVE_PREDICTION_GAUGE = Gauge('heart_disease_positive_predictions_total', 'Total prediksi positif penyakit jantung')


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Hanya catat metrik jika endpoint-nya /predict
    if request.url.path == "/predict":
        REQUEST_COUNTER.inc()
        LATENCY_HISTOGRAM.observe(process_time)
        
    return response


metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "Membangun_model", "saved_model", "heart_model.pkl")
)

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
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: HeartData):
    df = pd.DataFrame([data.dict()])
    prediction = int(model.predict(df)[0])
    
    if prediction == 1:
        POSITIVE_PREDICTION_GAUGE.inc()

    return {"prediction": prediction}
