# Step 1: vLLM Inference Service

## Описание

Развертывание инференс сервиса для большой языковой модели Qwen2.5-1.5B-Instruct с использованием vLLM.
vLLM обеспечивает высокую производительность благодаря PagedAttention и оптимизированному управлению памятью GPU.

## Структура

```
step1_vllm_inference/
├── src/
│   ├── __init__.py
│   └── llm_client.py          # Python клиент для vLLM API
├── notebooks/
│   └── chat_demo.ipynb        # Интерактивный notebook
├── docker-compose.yml         # Docker Compose для vLLM сервера
├── test_client.py            # Тестовый скрипт
├── pyproject.toml            # Poetry зависимости
├── Makefile                  # Команды автоматизации
└── README.md
```

## Требования

- Docker и Docker Compose
- NVIDIA GPU с драйверами (опционально, можно использовать CPU)
- Python 3.10+
- Poetry

## Установка

```bash
# Установка зависимостей
make install

# Или вручную
poetry install
```

## Запуск

### 1. Запуск vLLM сервера

```bash
make start
```

Сервер запустится на `http://localhost:8000`.

**Важно**: Первый запуск может занять несколько минут, так как модель будет скачана с HuggingFace (около 7GB).

Модель будет кэширована в `~/.cache/huggingface`.

### 2. Проверка логов

```bash
make logs
```

Дождитесь сообщения о готовности сервера.

### 3. Тестирование клиента

```bash
make test
```

Этот скрипт проверит:
- Состояние сервера (health check)
- Доступные модели
- Обычный chat completion
- Streaming chat completion
- Работу с русским языком

### 4. Интерактивный notebook

```bash
make notebook
```

Откройте `notebooks/chat_demo.ipynb` для интерактивного общения с моделью.

## Использование клиента

### Базовый пример

```python
from src.llm_client import VLLMClient

# Инициализация клиента
client = VLLMClient(base_url="http://localhost:8000/v1")

# Проверка здоровья
if not client.health_check():
    print("Server is not healthy!")
    exit(1)

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is MLOps?"}
]

response = client.chat_completion(
    messages=messages,
    temperature=0.7,
    max_tokens=512
)

print(response)
```

### Streaming

```python
# Streaming response
for chunk in client.chat_completion_stream(messages=messages):
    print(chunk, end="", flush=True)
```

## API Endpoints

vLLM предоставляет OpenAI-совместимое API:

- `GET /health` - Health check
- `GET /v1/models` - Список доступных моделей
- `POST /v1/chat/completions` - Chat completion
- `POST /v1/completions` - Text completion

## Конфигурация

Основные параметры в `docker-compose.yml`:

- `--model`: Модель HuggingFace (Qwen/Qwen2.5-1.5B-Instruct)
- `--max-model-len`: Максимальная длина контекста (4096 токенов)
- `--gpu-memory-utilization`: Процент использования GPU памяти (0.7)
- `--dtype half`: Использование FP16 для оптимизации памяти
- `--trust-remote-code`: Доверять коду модели

## Остановка и очистка

```bash
# Остановка сервера
make stop

# Полная очистка
make clean
```

## Особенности

1. **vLLM оптимизации**:
   - PagedAttention для эффективного управления KV-cache
   - Continuous batching для высокой throughput
   - Оптимизированные CUDA kernels

2. **OpenAI-совместимое API**:
   - Легко заменить OpenAI на vLLM
   - Поддержка streaming
   - Стандартные форматы запросов/ответов

3. **Модель Qwen2.5-1.5B-Instruct**:
   - Компактная модель (1.5B параметров)
   - Оптимизирована для эффективности
   - Поддержка множества языков, включая русский
   - Instruction-tuned для следования инструкциям
   - Использует FP16 для уменьшения памяти

## Troubleshooting

### Модель не загружается
- Проверьте интернет-соединение
- Убедитесь, что достаточно места на диске (~10GB)

### Ошибки GPU
- Для CPU режима измените `docker-compose.yml`, убрав секцию `deploy`
- Проверьте драйверы NVIDIA: `nvidia-smi`

### Сервер не отвечает
- Проверьте логи: `make logs`
- Дождитесь полной загрузки модели (может занять 1-2 минуты)
- Проверьте порт 8000: `lsof -i :8000`

## Дополнительная информация

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
