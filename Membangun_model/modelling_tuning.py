import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from heart_preprocessing import load_data, preprocess_data

def train_tuned_model():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../heart.csv'))
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, target_column='target')

    mlflow.set_experiment("Heart Disease Classification")

    # Hyperparameter Grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    with mlflow.start_run(run_name="Tuned_RandomForest"):
        predictions = best_model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Log Best Params & Metrics
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log Model Artifact
        mlflow.sklearn.log_model(best_model, "best_model")

        print(f"Tuned Model Trained! Best Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_tuned_model()
