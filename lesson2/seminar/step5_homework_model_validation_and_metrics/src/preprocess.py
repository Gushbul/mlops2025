import os

import pandas as pd


def preprocess_data():
    df = pd.read_csv("data/raw/tips.csv")

    df["high_tip"] = (df["tip"] / df["total_bill"]) > 0.2

    processed_df = df[["total_bill", "size", "high_tip"]].copy()

    os.makedirs("data/processed", exist_ok=True)
    processed_df.to_csv("data/processed/dataset.csv", index=False)


if __name__ == "__main__":
    preprocess_data()
