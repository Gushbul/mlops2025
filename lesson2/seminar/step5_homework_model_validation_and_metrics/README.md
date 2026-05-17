# Homework: Model Validation and Metrics

Базовый репозиторий: https://github.com/tam2511/mlops2025/tree/lesson1/lesson2/seminar/step4_ge_validation_data

## Задачи

### 1. Poetry и Pre-commit

Интегрируйте Poetry для управления зависимостями и настройте pre-commit хуки.

Включите в `.pre-commit-config.yaml`:
- black
- ruff
- isort
- end-of-file-fixer
- check-yaml

### 2. Обновление `src/evaluate.py`

Модифицируйте скрипт для записи метрик в `metrics/metrics.json`.

Метрики:
- accuracy
- количество строк в данных

### 3. Реализация `src/validate_model.py`

Создайте скрипт для проверки качества модели:
- Читает модель и данные
- Получает accuracy (пересчитывает или читает из `metrics/metrics.json`)
- Сравнивает с порогом `accuracy_min` из `params.yaml`
- Завершается с кодом != 0 при `accuracy < accuracy_min`

### 4. Обновление `dvc.yaml`

Модифицируйте стадии:

**evaluate**:
```yaml
metrics:
  - metrics/metrics.json
```

**validate_model** (новая стадия после evaluate):
```yaml
deps:
  - src/validate_model.py
  - models/model.pkl
  - params.yaml
```
