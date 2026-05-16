"""
Предобработка данных для обучения модели.
"""

import os
import sys
import pandas as pd


def preprocess_data(batch_number: int):
    """
    Предобработка данных для обучения.

    Args:
        batch_number: Номер версии датасета
    """

    df = pd.read_csv(f"data/processed/dataset_v{batch_number}.csv")

    df["high_tip"] = (df["tip"] > df["tip"].median()).astype(int)

    features = ["total_bill", "size", "high_tip"]
    df_processed = df[features].copy()

    df_processed = df_processed.dropna()

    os.makedirs("data/processed", exist_ok=True)

    df_processed.to_csv(
        f"data/processed/dataset_processed_v{batch_number}.csv", index=False
    )

    print(f"Предобработка завершена. Обработано {len(df_processed)} записей")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <batch_number>")
        sys.exit(1)

    batch_num = int(sys.argv[1])
    preprocess_data(batch_num)
