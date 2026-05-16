import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import BlipProcessor
from typing import Optional
import time


class ONNXImageCaptionService:
    """
    Сервис для инференса ONNX модели image captioning
    """

    def __init__(
        self, onnx_path: str, model_name: str = "Salesforce/blip-image-captioning-base"
    ):
        self.onnx_path = onnx_path
        self.model_name = model_name
        self.session: Optional[ort.InferenceSession] = None
        self.processor: Optional[BlipProcessor] = None
        self.loaded = False

    def load_model(self):
        """Загрузка ONNX модели и процессора"""
        if self.loaded:
            return

        print(f"Загрузка ONNX модели из {self.onnx_path}")

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)

        self.processor = BlipProcessor.from_pretrained(self.model_name)

        self.loaded = True
        print("ONNX модель и процессор загружены успешно")

    def preprocess_image(self, image: Image.Image) -> dict:
        """Предобработка изображения для ONNX модели"""

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(image, return_tensors="pt")

        image_input = inputs.pixel_values.numpy()

        token_id = getattr(self.processor.tokenizer, "bos_token_id", None)
        if token_id is None:
            token_id = getattr(self.processor.tokenizer, "cls_token_id", 101)

        input_ids = np.array([[token_id] * 16], dtype=np.int64)

        return {"image": image_input, "input_ids": input_ids}

    def predict(self, image: Image.Image) -> dict:
        """
        Выполнение ONNX инференса для одного изображения

        Args:
            image: PIL изображение

        Returns:
            dict с результатами инференса
        """
        if not self.loaded:
            raise ValueError("Модель не загружена. Вызовите load_model() сначала.")

        start_time = time.time()

        preprocess_start = time.time()
        onnx_inputs = self.preprocess_image(image)
        preprocess_time = time.time() - preprocess_start

        inference_start = time.time()
        try:

            caption = self._iterative_generation(onnx_inputs["image"], max_tokens=10)
            inference_time = time.time() - inference_start

            postprocess_start = time.time()
            if caption is None:
                caption = "Failed to generate caption"
                success = False
            else:
                success = True
            postprocess_time = time.time() - postprocess_start

        except Exception as e:
            inference_time = time.time() - inference_start
            postprocess_time = 0
            caption = f"ONNX inference error: {str(e)[:100]}"
            success = False

        total_time = time.time() - start_time

        return {
            "prediction": caption,
            "image_size": list(image.size),
            "model_type": "ONNX BLIP",
            "success": success,
            "timing": {
                "total_ms": total_time * 1000,
                "preprocess_ms": preprocess_time * 1000,
                "inference_ms": inference_time * 1000,
                "postprocess_ms": postprocess_time * 1000,
            },
            "onnx_details": {},
        }

    def predict_batch(self, images: list[Image.Image]) -> tuple[list[dict], dict]:
        """
        Batch инференс (выполняется последовательно для ONNX)

        Args:
            images: Список PIL изображений

        Returns:
            Tuple из списка результатов и статистики батча
        """
        results = []

        batch_start = time.time()
        for i, image in enumerate(images):
            result = self.predict(image)
            result["batch_index"] = i
            results.append(result)

        batch_time = time.time() - batch_start

        successful_requests = sum(1 for r in results if r.get("success", False))

        batch_stats = {
            "batch_size": len(images),
            "successful_requests": successful_requests,
            "failed_requests": len(images) - successful_requests,
            "total_batch_time_ms": batch_time * 1000,
            "avg_time_per_image_ms": batch_time * 1000 / len(images) if images else 0,
            "total_inference_time_ms": sum(
                r["timing"]["inference_ms"] for r in results
            ),
        }

        return results, batch_stats

    def _iterative_generation(
        self, image_input: np.ndarray, max_tokens: int = 10
    ) -> str:
        """
        Итеративная генерация текста с ONNX моделью
        """

        token_id = getattr(self.processor.tokenizer, "bos_token_id", None)
        if token_id is None:
            token_id = getattr(self.processor.tokenizer, "cls_token_id", 101)

        current_tokens = [token_id]
        generated_tokens = []

        for step in range(max_tokens):

            if len(current_tokens) < 16:
                input_ids = current_tokens + [token_id] * (16 - len(current_tokens))
            else:
                input_ids = current_tokens[-16:]

            input_ids_array = np.array([input_ids], dtype=np.int64)

            onnx_inputs = {"image": image_input, "input_ids": input_ids_array}

            try:
                outputs = self.session.run(None, onnx_inputs)
                logits = outputs[0]

                pos = len(current_tokens) - 1 if len(current_tokens) <= 16 else 15
                last_token_logits = logits[0, pos, :]
                predicted_id = int(np.argmax(last_token_logits))

                if predicted_id == 102:
                    break

                current_tokens.append(predicted_id)
                generated_tokens.append(predicted_id)

            except Exception:
                break

        if generated_tokens:
            try:
                return self.processor.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
            except Exception:
                return None
        return None
