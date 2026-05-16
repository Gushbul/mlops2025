"""
Управление батчами для автоматического пайплайна.
"""

import os
import json
import pandas as pd
from typing import Dict, Any
from prefect import task


@task
def get_next_batch_number() -> int:
    """Получение номера следующего батча для обработки."""
    state_file = "batch_state.json"

    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)
        next_batch = state.get("next_batch", 1)
    else:
        next_batch = 1

    print(f"Следующий батч для обработки: {next_batch}")
    return next_batch


@task
def update_batch_state(batch_number: int, metrics: Dict[str, Any]):
    """Обновление состояния батча."""
    state_file = "batch_state.json"

    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)
    else:
        state = {"processed_batches": [], "next_batch": 1}

    batch_info = {
        "batch_number": batch_number,
        "metrics": metrics,
        "timestamp": str(pd.Timestamp.now()),
    }

    state["processed_batches"].append(batch_info)
    state["next_batch"] = batch_number + 1

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    print(f"Обновлено состояние: следующий батч {state['next_batch']}")
    return state


@task
def check_max_batches_reached(batch_number: int, max_batches: int) -> bool:
    """Проверка достижения максимального количества батчей."""
    if batch_number > max_batches:
        print(f"Достигнуто максимальное количество батчей: {max_batches}")
        return True
    return False


@task
def check_data_availability(batch_number: int, batch_size: int) -> bool:
    """Проверка наличия данных для батча."""
    if not os.path.exists("data/raw/tips_full.csv"):
        print("Исходные данные не найдены")
        return False

    import pandas as pd

    df = pd.read_csv("data/raw/tips_full.csv")

    start_idx = (batch_number - 1) * batch_size
    available_data = len(df) - start_idx

    if available_data <= 0:
        print(f"Нет данных для батча {batch_number}")
        return False

    print(
        f"Доступно данных для батча {batch_number}: {min(available_data, batch_size)} записей"
    )
    return True
