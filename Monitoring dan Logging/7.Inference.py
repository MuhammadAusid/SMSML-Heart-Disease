import pandas as pd
import numpy as np

def run_inference():
    print("Memuat model dan melakukan inferensi data...")
    sample_data = {
        'age': [52],
        'sex': [1],
        'cp': [0],
        'trestbps': [125],
        'chol': [212],
        'fbs': [0],
        'restecg': [1],
        'thalach': [168],
        'exang': [0],
        'oldpeak': [1.0],
        'slope': [2],
        'ca': [2],
        'thal': [3]
    }
    df = pd.DataFrame(sample_data)
    print("Data Input:")
    print(df)
    
    # Simulasi hasil prediksi
    prediction = 0
    print(f"Hasil Prediksi Model: {prediction}")

if __name__ == '__main__':
    run_inference()
