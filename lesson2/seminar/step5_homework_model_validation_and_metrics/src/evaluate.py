import json
import os
import pickle

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

METRICS_PATH = "metrics/metrics.json"


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def evaluate_model():
    params = load_params()

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("data/processed/dataset.csv")

    x = df[["total_bill", "size"]]
    y = df["high_tip"]

    _, x_test, _, y_test = train_test_split(
        x, y, test_size=params["test_size"], random_state=params["seed"]
    )

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy),
        "rows": int(len(df)),
    }

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Rows: {metrics['rows']}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    evaluate_model()
