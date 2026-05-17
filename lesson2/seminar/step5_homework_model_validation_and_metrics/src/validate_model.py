import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

METRICS_PATH = Path("metrics/metrics.json")
MODEL_PATH = Path("models/model.pkl")
DATA_PATH = Path("data/processed/dataset.csv")
PARAMS_PATH = Path("params.yaml")


def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def compute_accuracy(model, df, params):
    x = df[["total_bill", "size"]]
    y = df["high_tip"]

    _, x_test, _, y_test = train_test_split(
        x, y, test_size=params["test_size"], random_state=params["seed"]
    )

    y_pred = model.predict(x_test)
    return float(accuracy_score(y_test, y_pred))


def read_accuracy_from_metrics():
    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    return float(metrics["accuracy"])


def validate_model():
    params = load_params()
    accuracy_min = float(params["accuracy_min"])

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(DATA_PATH)

    accuracy = read_accuracy_from_metrics()
    if accuracy is None:
        accuracy = compute_accuracy(model, df, params)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Minimum required accuracy: {accuracy_min:.4f}")

    if accuracy < accuracy_min:
        print(
            f"Model validation failed: accuracy {accuracy:.4f} "
            f"is below threshold {accuracy_min:.4f}"
        )
        sys.exit(1)

    print("Model validation passed")


if __name__ == "__main__":
    validate_model()
