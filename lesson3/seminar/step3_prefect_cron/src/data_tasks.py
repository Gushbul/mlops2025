"""
Задачи для работы с данными в Prefect.
"""

import os
import pandas as pd
import yaml
from prefect import task


@task
def load_params():
    """Загрузка параметров из params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


@task
def get_raw_data(url: str):
    """Загрузка исходных данных."""
    df = pd.read_csv(url)

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/tips_full.csv", index=False)

    print(f"Загружено {len(df)} записей")
    return len(df)


@task
def prepare_batch(batch_number: int, batch_size: int):
    """Подготовка батча данных."""
    df = pd.read_csv("data/raw/tips_full.csv")

    start_idx = (batch_number - 1) * batch_size
    end_idx = batch_number * batch_size

    batch_df = df.iloc[start_idx:end_idx].copy()

    if len(batch_df) == 0:
        print(f"Батч {batch_number} пуст")
        return 0

    os.makedirs("data/processed", exist_ok=True)
    batch_df.to_csv(f"data/processed/batch_{batch_number}.csv", index=False)

    print(f"Подготовлен батч {batch_number}: {len(batch_df)} записей")
    return len(batch_df)


@task
def merge_batches(batch_number: int, batch_size: int = None):
    """Объединение всех батчей до указанного номера."""
    all_data = []

    for i in range(1, batch_number + 1):
        batch_path = f"data/processed/batch_{i}.csv"
        if os.path.exists(batch_path):
            batch_df = pd.read_csv(batch_path)
            all_data.append(batch_df)
        else:
            print(f"Батч {i} не найден")

    if not all_data:
        return 0

    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df = merged_df.drop_duplicates().reset_index(drop=True)

    merged_df.to_csv(f"data/processed/dataset_v{batch_number}.csv", index=False)

    print(f"Создан датасет версии {batch_number}: {len(merged_df)} записей")
    return len(merged_df)


@task
def preprocess_data(batch_number: int, dataset_size: int = None):
    """Предобработка данных."""
    df = pd.read_csv(f"data/processed/dataset_v{batch_number}.csv")

    df["high_tip"] = (df["tip"] > df["tip"].median()).astype(int)

    features = ["total_bill", "size", "high_tip"]
    df_processed = df[features].copy().dropna()

    df_processed.to_csv(
        f"data/processed/dataset_processed_v{batch_number}.csv", index=False
    )

    print(f"Предобработка завершена: {len(df_processed)} записей")
    return len(df_processed)


@task
def create_dvc_tracking_file(processed_size: int = None):
    """Создание файла для отслеживания DVC."""
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/.dvc_tracked", "w") as f:
        f.write("DVC tracking file\n")
    return True
