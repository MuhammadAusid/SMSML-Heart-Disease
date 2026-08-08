
import os
import joblib
import pandas as pd


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "saved_model",
    "heart_model.pkl"
)

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


def load_model():
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    scaler = artifact["scaler"]

    return model, scaler


def predict_heart_disease(data: dict):

    model, scaler = load_model()

    input_data = pd.DataFrame(
        [data],
        columns=FEATURES
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    return {
        "prediction": int(prediction),
        "probability_no_disease": float(probability[0]),
        "probability_disease": float(probability[1])
    }


if __name__ == "__main__":

    sample_patient = {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 2,
        "thal": 3
    }

    result = predict_heart_disease(sample_patient)

    print("=== REAL INFERENCE ===")
    print(f"Prediction: {result['prediction']}")
    print(
        f"Probability no disease: "
        f"{result['probability_no_disease']:.4f}"
    )
    print(
        f"Probability disease: "
        f"{result['probability_disease']:.4f}"
    )
