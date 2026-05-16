# Шаг 1: DVC + MLflow пайплайн с инкрементальными данными

## Описание

Этот шаг демонстрирует создание ML пайплайна с использованием DVC для управления данными и экспериментами, и MLflow для логирования моделей и метрик. Основная особенность - работа с инкрементальными данными, имитирующими поступление новых данных с продакшена.

## Структура

```
step1_dvc_mlflow_pipeline/
├── data/
│   ├── raw/           # Исходные данные
│   └── processed/     # Обработанные данные по версиям
├── models/            # Сохраненные модели
├── metrics/           # Метрики модели
├── src/              # Исходный код
└── dvc.yaml          # Описание пайплайна DVC
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

3. Инициализация DVC:
```bash
dvc init --no-scm
```

4. Запуск MLflow UI (в отдельном терминале):
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

## Использование

**ВАЖНО:** Данная версия DVC не поддерживает параметр `--vars`. Вместо этого используйте Poetry для выполнения скриптов напрямую.

### Обучение первой модели (базовая версия)

```bash
# 1. Активация Poetry окружения (обязательно!)
poetry shell

# 2. Загрузка исходных данных
poetry run python src/get_data.py

# 3. Подготовка первого батча (первые 50 записей)
poetry run python src/prepare_batch.py 1

# 4. Объединение данных (пока только один батч)
poetry run python src/merge_data.py 1

# 5. Предобработка данных
poetry run python src/preprocess.py 1

# 6. Обучение модели (убедитесь, что MLflow UI запущен!)
poetry run python src/train.py 1

# 7. Оценка модели
poetry run python src/evaluate.py 1
```

### Добавление новых данных и переобучение

```bash
# Подготовка второго батча (записи 51-100)
poetry run python src/prepare_batch.py 2

# Объединение с предыдущими данными
poetry run python src/merge_data.py 2

# Предобработка объединенных данных
poetry run python src/preprocess.py 2

# Переобучение модели на расширенных данных
poetry run python src/train.py 2

# Оценка новой модели
poetry run python src/evaluate.py 2
```

### Полный пайплайн для батча (скрипт)

Создайте файл `run_batch.sh` для автоматизации:

```bash
#!/bin/bash
BATCH_NUMBER=$1

if [ -z "$BATCH_NUMBER" ]; then
    echo "Usage: ./run_batch.sh <batch_number>"
    exit 1
fi

echo "Запуск пайплайна для батча $BATCH_NUMBER..."

poetry run python src/prepare_batch.py $BATCH_NUMBER
poetry run python src/merge_data.py $BATCH_NUMBER
poetry run python src/preprocess.py $BATCH_NUMBER
poetry run python src/train.py $BATCH_NUMBER
poetry run python src/evaluate.py $BATCH_NUMBER

echo "Пайплайн завершен для батча $BATCH_NUMBER"
```

## Мониторинг

1. **MLflow UI**: Открыть http://localhost:5000 для просмотра экспериментов и моделей

2. **Метрики**: Файлы метрик сохраняются в папке `metrics/`

3. **Версионирование**: DVC отслеживает версии данных и моделей

## Устранение неполадок

### Проблема с подключением к MLflow

Если вы видите ошибку `Connection refused` при обучении модели:

1. **Проверьте, запущен ли MLflow сервер:**
   ```bash
   curl -s http://localhost:5000/health || echo "MLflow не запущен"
   ```

2. **Запустите MLflow сервер в отдельном терминале:**
   ```bash
   cd /path/to/step1_dvc_mlflow_pipeline
   poetry run mlflow ui --host 0.0.0.0 --port 5000
   ```

3. **Альтернативно, можно изменить настройки MLflow на локальное хранилище:**
   Временно измените в `params.yaml`:
   ```yaml
   mlflow:
     experiment_name: "incremental_training"
     tracking_uri: "./mlruns"  # Вместо http://localhost:5000
   ```

### Проблема с DVC переменными

Если DVC выдает ошибку `Could not find 'batch_number'`:
- Используйте прямой вызов скриптов через `poetry run python` вместо `dvc repro`
- Версия DVC в проекте не поддерживает параметр `--vars`

### Рекомендуемый порядок запуска

1. **Терминал 1** (MLflow сервер):
   ```bash
   cd step1_dvc_mlflow_pipeline
   poetry run mlflow ui --host 0.0.0.0 --port 5000
   ```

2. **Терминал 2** (пайплайн):
   ```bash
   cd step1_dvc_mlflow_pipeline
   poetry shell
   # Выполните команды пайплайна
   ```

## Демонстрация инкрементального обучения

Этот пайплайн показывает, как:

1. **Добавлять новые данные**: Каждый батч представляет новую порцию данных
2. **Объединять с историческими данными**: Все предыдущие батчи объединяются
3. **Переобучать модель**: Модель обучается на все больших объемах данных
4. **Сравнивать производительность**: MLflow позволяет сравнивать метрики разных версий
5. **Версионировать все артефакты**: DVC отслеживает данные, модели и метрики

Такой подход типичен для продакшен ML систем, где модель периодически дообучается на новых данных.
