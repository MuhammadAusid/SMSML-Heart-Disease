import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import modul preprocessing dari folder utama
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from heart_preprocessing import load_data, preprocess_data

def train_baseline_model():
    # Load & Preprocess Data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../heart.csv'))
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, target_column='target')

    # Set MLflow Experiment
    mlflow.set_experiment("Heart Disease Classification")

    with mlflow.start_run(run_name="Baseline_RandomForest"):
        # Hyperparameters
        n_estimators = 100
        max_depth = 5
        
        # Train Model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions & Metrics
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Logging Parameters & Metrics to MLflow
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Logging Artifact Model
        mlflow.sklearn.log_model(model, "model")

        print(f"Baseline Model Trained successfully! Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_baseline_model()
