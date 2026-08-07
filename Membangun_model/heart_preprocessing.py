import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df: pd.DataFrame, target_column: str = 'target'):
    """
    Melakukan pembersihan data, penanganan duplikat, 
    pemisahan fitur & label, serta scaling data.
    """
    
    df = df.drop_duplicates()
    
    df = df.dropna()
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
