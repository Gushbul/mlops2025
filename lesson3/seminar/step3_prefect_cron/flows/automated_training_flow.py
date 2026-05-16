"""
Автоматизированный поток обучения с cron триггером.
"""

import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from prefect import flow
from data_tasks import (
    load_params,
    get_raw_data,
    prepare_batch,
    merge_batches,
    preprocess_data,
    create_dvc_tracking_file,
)
from model_tasks import train_model, evaluate_model
from batch_manager import (
    get_next_batch_number,
    update_batch_state,
    check_max_batches_reached,
    check_data_availability,
)


@flow(name="Automated ML Pipeline", log_prints=True)
def automated_training_pipeline():
    """
    Автоматизированный поток ML пайплайна с управлением батчами.
    """
    print("Запуск автоматизированного пайплайна")

    params = load_params()

    batch_number = get_next_batch_number()

    max_batches_reached = check_max_batches_reached(
        batch_number, params["automation"]["max_batches"]
    )

    if max_batches_reached:
        print("Достигнуто максимальное количество батчей. Останавливаем.")
        return {"status": "max_batches_reached", "batch_number": batch_number}

    raw_data_path = "data/raw/tips_full.csv"
    if not os.path.exists(raw_data_path):
        get_raw_data(params["data"]["url"])

    data_available = check_data_availability(batch_number, params["data"]["batch_size"])

    if not data_available:
        print(f"Нет данных для батча {batch_number}")
        return {"status": "no_data", "batch_number": batch_number}

    batch_size = prepare_batch(batch_number, params["data"]["batch_size"])

    if batch_size == 0:
        print("Подготовлен пустой батч")
        return {"status": "empty_batch", "batch_number": batch_number}

    dataset_size = merge_batches(batch_number, batch_size)

    processed_size = preprocess_data(batch_number, dataset_size)

    dvc_tracking = create_dvc_tracking_file(processed_size)

    train_metrics = train_model(batch_number, params, processed_size, dvc_tracking)

    eval_metrics = evaluate_model(batch_number, params, train_metrics)

    all_metrics = {
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "processed_size": processed_size,
        "dataset_size": dataset_size,
    }

    update_batch_state(batch_number, all_metrics)

    print(f"Автоматический пайплайн завершен для батча {batch_number}")
    print(f"Обработано данных: {processed_size}")
    print(f"Метрики: {all_metrics}")

    return {"status": "success", "batch_number": batch_number, "metrics": all_metrics}


@flow(name="Manual Training Pipeline", log_prints=True)
def manual_training_pipeline(batch_number: int):
    """
    Ручной запуск пайплайна для конкретного батча.

    Args:
        batch_number: Номер батча для обучения
    """
    print(f"Ручной запуск пайплайна для батча {batch_number}")

    params = load_params()

    raw_data_path = "data/raw/tips_full.csv"
    if not os.path.exists(raw_data_path):
        get_raw_data(params["data"]["url"])

    data_available = check_data_availability(batch_number, params["data"]["batch_size"])

    if not data_available:
        print(f"Нет данных для батча {batch_number}")
        return {"status": "no_data", "batch_number": batch_number}

    batch_size = prepare_batch(batch_number, params["data"]["batch_size"])
    dataset_size = merge_batches(batch_number, batch_size)
    processed_size = preprocess_data(batch_number, dataset_size)
    dvc_tracking = create_dvc_tracking_file(processed_size)

    train_metrics = train_model(batch_number, params, processed_size, dvc_tracking)
    eval_metrics = evaluate_model(batch_number, params, train_metrics)

    all_metrics = {
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "processed_size": processed_size,
    }

    print(f"Ручной пайплайн завершен для батча {batch_number}")
    return {"status": "success", "batch_number": batch_number, "metrics": all_metrics}


if __name__ == "__main__":
    if len(sys.argv) > 1:

        batch_num = int(sys.argv[1])
        manual_training_pipeline(batch_num)
    else:

        automated_training_pipeline()
