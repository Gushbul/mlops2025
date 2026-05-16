"""
Загрузка исходных данных.
"""

import os
import pandas as pd
import yaml


def load_params():
    """Загрузка параметров из params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def get_data():
    """Загрузка данных из внешнего источника."""
    params = load_params()

    df = pd.read_csv(params["data"]["url"])

    os.makedirs("data/raw", exist_ok=True)

    df.to_csv("data/raw/tips_full.csv", index=False)

    print(f"Загружено {len(df)} записей")


if __name__ == "__main__":
    get_data()
