"""
Подготовка батча данных для инкрементального обучения.
"""

import os
import sys
import pandas as pd
import yaml


def load_params():
    """Загрузка параметров из params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def prepare_batch(batch_number: int):
    """
    Подготовка батча данных.

    Args:
        batch_number: Номер батча (начиная с 1)
    """
    params = load_params()
    batch_size = params["data"]["batch_size"]

    df = pd.read_csv("data/raw/tips_full.csv")

    start_idx = (batch_number - 1) * batch_size
    end_idx = batch_number * batch_size

    batch_df = df.iloc[start_idx:end_idx].copy()

    if len(batch_df) == 0:
        print(f"Батч {batch_number} пуст, данные закончились")
        return

    os.makedirs("data/processed", exist_ok=True)

    batch_df.to_csv(f"data/processed/batch_{batch_number}.csv", index=False)

    print(f"Подготовлен батч {batch_number}: {len(batch_df)} записей")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_batch.py <batch_number>")
        sys.exit(1)

    batch_num = int(sys.argv[1])
    prepare_batch(batch_num)
