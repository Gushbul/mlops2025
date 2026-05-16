"""
Обучение модели с логированием в MLflow.
"""

import json
import os
import pickle
import sys
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def load_params():
    """Загрузка параметров из params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def train_model(batch_number: int):
    """
    Обучение модели с логированием в MLflow.

    Args:
        batch_number: Номер версии данных
    """
    params = load_params()

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

    with mlflow.start_run(run_name=f"model_v{batch_number}"):

        mlflow.log_params(
            {
                "batch_number": batch_number,
                "data_size": len(df),
                "test_size": params["model"]["test_size"],
                "seed": params["model"]["seed"],
            }
        )

        model = LogisticRegression(random_state=params["model"]["seed"])
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

        mlflow.sklearn.log_model(model, f"model_v{batch_number}")

        os.makedirs("models", exist_ok=True)
        os.makedirs("metrics", exist_ok=True)

        with open(f"models/model_v{batch_number}.pkl", "wb") as f:
            pickle.dump(model, f)

        metrics_data = {
            "accuracy": accuracy,
            "f1_score": f1,
            "data_size": len(df),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        with open(f"metrics/metrics_v{batch_number}.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

        print(f"Модель v{batch_number} обучена:")
        print(f"  Точность: {accuracy:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  Размер данных: {len(df)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <batch_number>")
        sys.exit(1)

    batch_num = int(sys.argv[1])
    train_model(batch_num)
