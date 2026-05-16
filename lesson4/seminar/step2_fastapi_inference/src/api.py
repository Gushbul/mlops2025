from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
from typing import List
import os

from .model_service import ONNXImageCaptionService


app = FastAPI(
    title="ONNX Image Captioning Service",
    description="FastAPI сервис для инференса ONNX модели генерации описаний изображений",
    version="1.0.0",
)


model_service: ONNXImageCaptionService = None


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске сервиса"""
    global model_service

    onnx_path = "models/blip_model.onnx"

    if not os.path.exists(onnx_path):
        print(f"⚠️  ONNX модель не найдена: {onnx_path}")
        print(
            "Скопируйте модель из step1: cp ../step1_onnx_model/models/blip_model.onnx models/"
        )
        model_service = None
        return

    try:
        model_service = ONNXImageCaptionService(onnx_path)
        model_service.load_model()
        print("✅ ONNX модель загружена успешно")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        model_service = None


@app.get("/")
async def root():
    """Проверка работы сервиса"""
    return {
        "message": "ONNX Image Captioning Service",
        "status": "running",
        "model_loaded": model_service is not None,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model_service is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": model_service.model_name,
        "onnx_path": model_service.onnx_path,
    }


def validate_image(file: UploadFile) -> Image.Image:
    """Валидация и загрузка изображения"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    try:
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Ошибка обработки изображения: {str(e)}"
        )


@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    """
    Инференс для одного изображения
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:

        image = validate_image(file)

        result = model_service.predict(image)

        return {
            "success": True,
            "filename": file.filename,
            "image_size": image.size,
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Batch инференс для нескольких изображений
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    if len(files) == 0:
        raise HTTPException(
            status_code=400, detail="Необходимо загрузить хотя бы одно изображение"
        )

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Максимум 10 изображений за раз")

    try:

        images = []
        filenames = []

        for file in files:
            image = validate_image(file)
            images.append(image)
            filenames.append(file.filename)

        results, batch_stats = model_service.predict_batch(images)

        for i, result in enumerate(results):
            result["filename"] = filenames[i]
            result["image_size"] = list(images[i].size)

        return {"success": True, "batch_stats": batch_stats, "results": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка batch инференса: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """
    Получение метрик сервиса
    """
    if model_service is None:
        return {"model_loaded": False}

    return {
        "model_loaded": True,
        "model_name": model_service.model_name,
        "model_type": "ONNX BLIP",
        "onnx_path": model_service.onnx_path,
    }
