# Шаг 2: Prefect для оркестрации пайплайна

## Описание

В этом шаге мы переходим от использования DVC как основного оркестратора к Prefect. DVC используется только для версионирования данных, а вся логика выполнения пайплайна описана в Prefect flows и tasks.

## Структура

```
step2_prefect_orchestration/
├── data/              # Данные (версионируются DVC)
├── models/            # Модели
├── metrics/           # Метрики
├── src/              # Задачи Prefect
│   ├── data_tasks.py     # Задачи для работы с данными
│   └── model_tasks.py    # Задачи для работы с моделями
├── flows/            # Потоки Prefect
│   └── training_flow.py  # Основной поток обучения
└── dvc.yaml          # Только для версионирования
```

## Установка и настройка

1. Установка зависимостей:
```bash
poetry install
```

2. Активация виртуального окружения:
```bash
poetry shell
```

3. **Опционально** - Запуск Prefect server (в отдельном терминале):
```bash
poetry run prefect server start
```
*Примечание: Если сервер не запущен, Prefect автоматически создаст временный сервер*

4. Запуск MLflow UI (в отдельном терминале):
```bash
poetry run mlflow ui --host 0.0.0.0 --port 5000
```

5. Открыть Prefect UI: http://localhost:4200 (если сервер запущен)

## Использование

### Запуск пайплайна через Python (рекомендуемый способ)

```bash
# Обучение первой модели (50 записей)
poetry run python flows/training_flow.py 1

# Обучение второй модели с дополнительными данными (100 записей)
poetry run python flows/training_flow.py 2

# Обучение третьей модели (150 записей)
poetry run python flows/training_flow.py 3
```

### Быстрый запуск через скрипт

```bash
# Использование автоматизированного скрипта
./run_prefect_batch.sh 1
./run_prefect_batch.sh 2
```

### Запуск пайплайна через Prefect CLI

```bash
# Регистрация потока
prefect deployment build flows/training_flow.py:training_pipeline -n "ml-pipeline"

# Применение деплоймента
prefect deployment apply training_pipeline-deployment.yaml

# Запуск конкретного батча
prefect deployment run "ML Training Pipeline/ml-pipeline" --param batch_number=1
```

### Версионирование данных с DVC

```bash
# Инициализация DVC (если еще не сделано)
dvc init --no-scm

# Отслеживание версий данных
dvc add data/processed/

# Коммит изменений в данных
dvc commit
```

## Преимущества Prefect

1. **Визуализация**: Графическое отображение пайплайна в UI
2. **Мониторинг**: Отслеживание статуса выполнения задач
3. **Обработка ошибок**: Автоматические повторы и обработка сбоев
4. **Параллелизм**: Возможность параллельного выполнения задач
5. **Планирование**: Встроенная поддержка cron-подобного планирования

## Архитектура

- **Tasks**: Атомарные функции для выполнения конкретных операций
- **Flows**: Композиция задач в логическую последовательность
- **Deployments**: Конфигурация для запуска потоков
- **Work Queues**: Очереди для распределения работы

## Мониторинг

1. **Prefect UI**: http://localhost:4200 - статус выполнения пайплайнов (если запущен сервер)
2. **MLflow UI**: http://localhost:5000 - эксперименты и модели
3. **Логи**: Детальные логи выполнения в консоли и Prefect UI
4. **Временный сервер**: Prefect автоматически создает временный сервер для отслеживания

## Устранение неполадок

### Проблема с подключением к MLflow

Если видите ошибку подключения к MLflow:

1. **Проверьте статус MLflow сервера:**
   ```bash
   curl -s http://localhost:5000/health || echo "MLflow не запущен"
   ```

2. **Запустите MLflow в отдельном терминале:**
   ```bash
   cd /path/to/step2_prefect_orchestration
   poetry run mlflow ui --host 0.0.0.0 --port 5000
   ```

3. **Альтернативно, используйте локальное хранилище:**
   Измените в `params.yaml`:
   ```yaml
   mlflow:
     experiment_name: "prefect_pipeline"
     tracking_uri: "./mlruns"  # Вместо http://localhost:5000
   ```

### Предупреждения Pydantic

Предупреждения о `pyproject_toml_table_header` и `toml_file` в Pydantic безопасны и не влияют на работу пайплайна.

### Рекомендуемый порядок запуска

1. **Терминал 1** (MLflow, обязательно):
   ```bash
   cd step2_prefect_orchestration
   poetry run mlflow ui --host 0.0.0.0 --port 5000
   ```

2. **Терминал 2** (Prefect Server, опционально):
   ```bash
   cd step2_prefect_orchestration
   poetry run prefect server start
   ```

3. **Терминал 3** (пайплайн):
   ```bash
   cd step2_prefect_orchestration
   poetry run python flows/training_flow.py 1
   ```

### Преимущества по сравнению с DVC

- ✅ Автоматическое управление зависимостями между задачами
- ✅ Веб-интерфейс для мониторинга выполнения
- ✅ Обработка ошибок и повторные запуски
- ✅ Параллельное выполнение независимых задач
- ✅ Детальные логи и трассировка выполнения
- ✅ Возможность планирования и автоматизации
