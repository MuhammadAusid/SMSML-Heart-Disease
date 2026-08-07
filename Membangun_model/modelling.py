
import os
import sys
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Import preprocessing
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from heart_preprocessing import load_data, preprocess_data


def train_baseline_model():


    data_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../heart.csv"
        )
    )

    df = load_data(data_path)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        df,
        target_column="target"
    )


    mlflow.set_experiment("Heart Disease Classification")

    mlflow.autolog()

    with mlflow.start_run(run_name="Baseline_RandomForest"):

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction)
        recall = recall_score(y_test, prediction)
        f1 = f1_score(y_test, prediction)

        print("=" * 50)
        print("Baseline Random Forest")
        print("=" * 50)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("=" * 50)
        print("Training selesai.")
        print("=" * 50)


if __name__ == "__main__":
    train_baseline_model()
