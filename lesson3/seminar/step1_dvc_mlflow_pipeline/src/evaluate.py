"""
Оценка обученной модели.
"""

import json
import pickle
import sys
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


def load_params():
    """Загрузка параметров из params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def evaluate_model(batch_number: int):
    """
    Оценка модели на тестовых данных.

    Args:
        batch_number: Номер версии модели
    """
    params = load_params()

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

    print(f"Оценка модели v{batch_number}:")
    print(f"  Тестовая точность: {accuracy:.4f}")
    print(f"  Тестовый F1-score: {f1:.4f}")
    print(f"  Количество тестовых образцов: {len(X_test)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <batch_number>")
        sys.exit(1)

    batch_num = int(sys.argv[1])
    evaluate_model(batch_num)
