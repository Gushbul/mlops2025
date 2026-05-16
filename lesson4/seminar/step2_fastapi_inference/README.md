# Шаг 2: FastAPI сервис для ONNX инференса

## Описание

FastAPI веб-сервис для инференса ONNX модели image captioning с поддержкой single и batch обработки изображений.

## Структура

```
step2_fastapi_inference/
├── src/
│   ├── __init__.py
│   ├── api.py              # FastAPI приложение
│   └── model_service.py    # Сервис ONNX модели
├── models/                 # Папка для ONNX модели (копировать из step1)
├── test_images/           # Тестовые изображения
├── main.py                # Запуск сервера
├── client_test.py         # Тестовый клиент
└── pyproject.toml        # Зависимости Poetry
```

## Установка

```bash
poetry install
poetry shell
```

## Подготовка

Скопируйте ONNX модель из step1:
```bash
cp ../step1_onnx_model/models/blip_model.onnx models/
```

## Запуск сервиса

```bash
python main.py
```

Сервис будет доступен по адресам:
- API: http://localhost:8000
- Документация: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Тестирование

В отдельном терминале:
```bash
python client_test.py
```

## API Endpoints

### GET /health
Проверка состояния сервиса

### POST /predict
Инференс одного изображения
- Принимает: multipart/form-data с изображением
- Возвращает: результат инференса с метриками времени

### POST /predict_batch
Batch инференс нескольких изображений
- Принимает: multipart/form-data с несколькими изображениями (до 10)
- Возвращает: результаты всех изображений + статистика батча

### GET /metrics
Метрики сервиса

## Особенности

- Инференс только на CPU (batch_size=1)
- Подробные метрики времени (preprocess, inference, postprocess)
- Валидация входных данных
- Обработка ошибок
- Автоматическая документация API
