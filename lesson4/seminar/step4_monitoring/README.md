# Шаг 4: Мониторинг FastAPI сервиса - Домашнее задание

## Задание

Реализовать систему мониторинга для FastAPI сервиса из шага 2 с настраиваемыми алертами и цветным логированием.

## Требования к реализации

### 1. Интеграция с FastAPI сервисом
- Использовать сервис инференса ONNX модели из шага 2
- Запускать FastAPI сервис программно или подключаться к запущенному
- Мониторить endpoints: `/health` и `/predict`

### 2. Мониторинг основных метрик
- **Response Time**: время отклика API запросов
- **P95 Latency**: 95-й перцентиль времени ответа
- **Error Rate**: процент неудачных запросов
- **Health Status**: статус работоспособности сервиса
- **Consecutive Failures**: количество последовательных ошибок

### 3. Тестирование инференса
- Отправлять POST запросы с изображениями на `/predict`
- Поддерживать любые форматы изображений (jpg, png)
- Логировать время обработки и результат предсказания
- Проверять корректность ответа API

### 4. Цветные алерты
Реализовать систему алертов с цветовой индикацией:

- **Зеленый**: Нормальная работа (метрики в пределах нормы)
- **Желтый**: Предупреждение (превышены warning пороги)
- **Красный**: Критическое состояние (превышены critical пороги)

### 5. Конфигурируемые пороговые значения

Пример конфигурационного файла `monitoring_config.yaml`:

```yaml
service:
  host: "localhost"
  port: 8000
  base_url: "http://localhost:8000"

monitoring:
  check_interval_seconds: 30
  samples_per_check: 3
  request_timeout_seconds: 10

thresholds:
  response_time_ms:
    warning: 2000
    critical: 5000
  p95_latency_ms:
    warning: 3000
    critical: 6000
  error_rate_percent:
    warning: 10
    critical: 25
  consecutive_failures:
    warning: 3
    critical: 5

alerts:
  enabled: true
  cooldown_minutes: 5

logging:
  console_colors: true
  log_file: "logs/monitoring.log"
  metrics_file: "logs/metrics.jsonl"
```

### 6. Логирование
- Структурированные логи в JSON формате
- Отдельный файл для метрик в JSONL формате
- Консольный вывод с цветовым кодированием
- Временные метки для всех событий

## Технические требования

### Архитектура системы
```
step4_monitoring/
├── main.py                    # Точка входа
├── src/
│   ├── monitor.py            # Основная логика мониторинга
│   ├── logger.py             # Цветное логирование
│   └── config.py             # Работа с конфигурацией
├── config/
│   └── monitoring_config.yaml
├── logs/                     # Создается автоматически
└── README.md
```
