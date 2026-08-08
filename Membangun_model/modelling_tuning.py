
import os
import sys
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from heart_preprocessing import load_data, preprocess_data


def train_tuned_model():

    data_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../heart.csv"
        )
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {data_path}"
        )

    df = load_data(data_path)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        df,
        target_column="target"
    )

    mlflow.set_experiment("Heart Disease Classification")
    mlflow.autolog()

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 10],
        "criterion": ["gini", "entropy"]
    }

    base_model = RandomForestClassifier(
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    prediction = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    print("=" * 50)
    print("Hyperparameter Tuning")
    print("=" * 50)

    print("Best Parameters")
    print(grid_search.best_params_)

    print()

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("=" * 50)
    print("Training selesai.")
    print("=" * 50)

    model_dir = os.path.join(
        os.path.dirname(__file__),
        "saved_model"
    )

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        "heart_model.pkl"
    )

    # Simpan model dan scaler untuk kebutuhan inference
    model_artifact = {
        "model": best_model,
        "scaler": scaler
    }

    joblib.dump(
        model_artifact,
        model_path
    )

    print("Model dan scaler berhasil disimpan pada:")
    print(model_path)


if __name__ == "__main__":
    train_tuned_model()
