"""
Объединение текущего батча с предыдущими данными.
"""

import os
import sys
import pandas as pd


def merge_data(batch_number: int):
    """
    Объединение данных всех батчей до указанного номера.

    Args:
        batch_number: Номер текущего батча
    """
    all_data = []

    for i in range(1, batch_number + 1):
        batch_path = f"data/processed/batch_{i}.csv"
        if os.path.exists(batch_path):
            batch_df = pd.read_csv(batch_path)
            all_data.append(batch_df)
            print(f"Добавлен батч {i}: {len(batch_df)} записей")
        else:
            print(f"Батч {i} не найден, пропускаем")

    if not all_data:
        print("Нет данных для объединения")
        return

    merged_df = pd.concat(all_data, ignore_index=True)

    merged_df = merged_df.drop_duplicates().reset_index(drop=True)

    os.makedirs("data/processed", exist_ok=True)

    merged_df.to_csv(f"data/processed/dataset_v{batch_number}.csv", index=False)

    print(f"Создан датасет версии {batch_number}: {len(merged_df)} записей")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python merge_data.py <batch_number>")
        sys.exit(1)

    batch_num = int(sys.argv[1])
    merge_data(batch_num)
