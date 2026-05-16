import pickle
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def evaluate_model():
    params = load_params()

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("data/processed/dataset.csv")

    X = df[["total_bill", "size"]]
    y = df["high_tip"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["seed"]
    )

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    evaluate_model()
