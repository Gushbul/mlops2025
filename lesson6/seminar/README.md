# Семинар: LLMOps - RAG и Function Calling

Практические задания по изучению операционализации больших языковых моделей.

## Цели семинара

1. Научиться развертывать LLM инференс сервисы с vLLM
2. Реализовать клиент для взаимодействия с моделью
3. Освоить function calling для интеграции с внешними API
4. Создать интерактивного агента с доступом к инструментам

## Структура

- **step1_vllm_inference/**: Развертывание vLLM сервиса
  - vLLM сервер с Qwen3-4B-Instruct
  - Python клиент для API
  - Jupyter notebook для интерактивного общения
  - Docker Compose для запуска

- **step2_function_calling/**: Function calling с визуализацией поз
  - Pose Visualization API (визуализация поз человека)
  - Агент с function calling
  - Генерация структурированных данных из текста
  - Создание изображений через API
  - Notebook для демонстрации

## Требования

- Python 3.10+
- Poetry для управления зависимостями
- Docker и Docker Compose
- GPU с минимум 8GB VRAM (рекомендуется) или CPU
- Make для автоматизации команд

## Быстрый старт

```bash
# Step 1: vLLM инференс
cd step1_vllm_inference
make install      # установка зависимостей
make start        # запуск vLLM сервиса
make notebook     # запуск Jupyter
make test         # тестирование клиента

# Step 2: Function calling
cd step2_function_calling
make install      # установка зависимостей
make start-all    # запуск всех сервисов
make notebook     # запуск Jupyter
make test         # тестирование агента
```

## Последовательность выполнения

1. Начните с `step1_vllm_inference` для настройки базового инференса
2. Перейдите к `step2_function_calling` для добавления function calling
3. Экспериментируйте с параметрами в notebooks
