import os
import pickle
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def train_model():
    params = load_params()

    df = pd.read_csv("data/processed/dataset.csv")

    X = df[["total_bill", "size"]]
    y = df["high_tip"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["seed"]
    )

    model = LogisticRegression(random_state=params["seed"])
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_model()
