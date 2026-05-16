# Step 3: Active Learning with Prefect and MLflow

Реализация Active Learning с использованием uncertainty sampling, Prefect для оркестрации и MLflow для трекинга экспериментов.

## Описание

В этом шаге мы:
1. Реализуем Active Learning с uncertainty sampling
2. Начинаем с 10% помеченных данных
3. На каждой итерации используем модель k-1 для выбора наиболее неопределенных примеров
4. Добавляем выбранные примеры к обучающей выборке
5. Используем Prefect для оркестрации и MLflow для трекинга
6. Сравниваем эффективность с обычным Continuous Training

## Структура

- `src/` - основные модули
  - `data_manager.py` - управление данными для Active Learning
  - `model_trainer.py` - обучение и оценка модели
  - `mlflow_tracker.py` - интеграция с MLflow
  - `active_learning.py` - uncertainty sampling и AL логика
- `flows/` - Prefect flows
  - `active_learning_flow.py` - основной flow для Active Learning
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

# Запуск Active Learning flow
make run-al

# Запуск с разными стратегиями uncertainty sampling
make run-al-entropy    # entropy-based (default)
make run-al-margin     # margin-based
make run-al-confident  # least confident

# Просмотр результатов в MLflow UI
make mlflow-ui

# Очистка результатов
make clean
```

## Результаты

После выполнения создаются:
- `models/` - модели для каждой AL итерации
- `metrics/` - метрики для каждой итерации
- `mlruns/` - MLflow эксперименты с метриками
- Логи выполнения Prefect flows

## Стратегии Uncertainty Sampling

1. **Entropy** (по умолчанию) - выбирает примеры с максимальной энтропией предсказаний
2. **Margin** - выбирает примеры с минимальной разностью между двумя лучшими предсказаниями
3. **Least Confident** - выбирает примеры с минимальной уверенностью в лучшем предсказании

## Преимущества Active Learning

- Более эффективное использование данных
- Быстрое достижение хорошего качества с меньшим количеством помеченных примеров
- Адаптивный выбор наиболее информативных образцов
