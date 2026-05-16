# Step 2: Continuous Training with Prefect and MLflow

Реализация Continuous Training с использованием Prefect для оркестрации и MLflow для трекинга экспериментов.

## Описание

В этом шаге мы:
1. Создаем валидационный датасет (отдельно от test)
2. Реализуем Prefect flow для постепенного обучения
3. Запускаем 10 итераций, каждый раз добавляя 10% данных
4. Используем MLflow для трекинга всех экспериментов
5. Сравниваем метрики на валидации и тесте

## Структура

- `src/` - основные модули
  - `data_manager.py` - управление данными и разделением
  - `model_trainer.py` - обучение и оценка модели
  - `mlflow_tracker.py` - интеграция с MLflow
- `flows/` - Prefect flows
  - `continuous_training_flow.py` - основной flow CT
- `pyproject.toml` - зависимости
- `Makefile` - команды для запуска
- `models/` - сохраненные модели (создается автоматически)
- `mlruns/` - MLflow tracking (создается автоматически)

## Использование

```bash
# Установка зависимостей
make install

# Запуск Prefect сервера (в отдельном терминале)
make prefect-server

# Запуск Continuous Training flow
make run-ct

# Просмотр результатов в MLflow UI
make mlflow-ui

# Очистка результатов
make clean
```

## Результаты

После выполнения создаются:
- `models/` - модели для каждой итерации
- `mlruns/` - MLflow эксперименты с метриками
- Логи выполнения Prefect flows
