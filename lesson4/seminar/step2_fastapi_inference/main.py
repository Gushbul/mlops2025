import uvicorn


def main():
    """Запуск FastAPI сервиса"""
    print("🚀 Запуск ONNX Image Captioning Service")
    print("📡 Сервис будет доступен по адресу: http://localhost:8000")
    print("📚 Документация API: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/health")

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")


if __name__ == "__main__":
    main()
