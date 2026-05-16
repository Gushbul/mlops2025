"""
Задачи для работы с моделями в Prefect.
"""

import json
import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from prefect import task
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


@task
def train_model(
    batch_number: int,
    params: dict,
    processed_size: int = None,
    dvc_tracking: bool = None,
):
    """Обучение модели с логированием в MLflow."""
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    df = pd.read_csv(f"data/processed/dataset_processed_v{batch_number}.csv")

    X = df[["total_bill", "size"]]
    y = df["high_tip"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["seed"],
    )

    with mlflow.start_run(run_name=f"prefect_model_v{batch_number}"):
        mlflow.log_params(
            {
                "batch_number": batch_number,
                "data_size": len(df),
                "test_size": params["model"]["test_size"],
                "seed": params["model"]["seed"],
                "orchestrator": "prefect",
            }
        )

        model = LogisticRegression(random_state=params["model"]["seed"])
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

        mlflow.sklearn.log_model(model, f"prefect_model_v{batch_number}")

        os.makedirs("models", exist_ok=True)

        with open(f"models/model_v{batch_number}.pkl", "wb") as f:
            pickle.dump(model, f)

        os.makedirs("metrics", exist_ok=True)

        metrics_data = {
            "accuracy": accuracy,
            "f1_score": f1,
            "data_size": len(df),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        with open(f"metrics/metrics_v{batch_number}.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

        print(f"Модель v{batch_number} обучена (Prefect):")
        print(f"  Точность: {accuracy:.4f}")
        print(f"  F1-score: {f1:.4f}")

        return {"accuracy": accuracy, "f1_score": f1, "data_size": len(df)}


@task
def evaluate_model(batch_number: int, params: dict, train_metrics: dict = None):
    """Оценка модели."""
    df = pd.read_csv(f"data/processed/dataset_processed_v{batch_number}.csv")

    X = df[["total_bill", "size"]]
    y = df["high_tip"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["model"]["test_size"],
        random_state=params["model"]["seed"],
    )

    with open(f"models/model_v{batch_number}.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True)

    test_metrics = {
        "test_accuracy": accuracy,
        "test_f1_score": f1,
        "classification_report": report,
        "test_samples": len(X_test),
    }

    with open(f"metrics/test_metrics_v{batch_number}.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Оценка модели v{batch_number} (Prefect):")
    print(f"  Тестовая точность: {accuracy:.4f}")
    print(f"  Тестовый F1-score: {f1:.4f}")

    return {"test_accuracy": accuracy, "test_f1_score": f1}
