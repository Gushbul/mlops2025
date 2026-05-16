import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from transformers import BlipProcessor


class ONNXModelTester:
    """
    Тестер ONNX модели BLIP
    """

    def __init__(
        self, onnx_path: str, model_name: str = "Salesforce/blip-image-captioning-base"
    ):
        self.onnx_path = onnx_path
        self.model_name = model_name
        self.session = None
        self.processor = None

    def load_onnx_model(self):
        """Загрузка ONNX модели"""
        print(f"Загрузка ONNX модели из {self.onnx_path}")

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)

        self.processor = BlipProcessor.from_pretrained(self.model_name)

        print("ONNX модель загружена успешно")

    def test_inference(self, image_url: str = None):
        """Тестирование инференса ONNX модели"""
        if self.session is None:
            raise ValueError(
                "ONNX модель не загружена. Вызовите load_onnx_model() сначала."
            )

        if image_url is None:
            image_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"

        print(f"Тестирование ONNX модели на изображении: {image_url}")
        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        inputs = self.processor(raw_image, return_tensors="pt")

        image_input = inputs.pixel_values.numpy()

        bos_id = getattr(self.processor.tokenizer, "bos_token_id", None)
        if bos_id is None:
            bos_id = getattr(self.processor.tokenizer, "cls_token_id", 101)

        print(f"Используем token_id: {bos_id}")

        input_ids = np.array([[bos_id] * 16], dtype=np.int64)

        print("Подготовленные входы для ONNX:")
        print(f"  - image: {image_input.shape} {image_input.dtype}")
        print(f"  - input_ids: {input_ids.shape} {input_ids.dtype}")

        ort_inputs = {"image": image_input, "input_ids": input_ids}

        print("Запуск ONNX инференса...")
        try:
            ort_outputs = self.session.run(None, ort_inputs)
            print("✅ ONNX инференс завершен успешно!")

            print(f"Количество выходов: {len(ort_outputs)}")
            for i, output in enumerate(ort_outputs):
                print(f"  Выход {i}: {output.shape} {output.dtype}")

            logits = ort_outputs[0]

            if len(logits.shape) == 3 and logits.shape[2] == 30524:

                last_token_logits = logits[0, -1, :]
                predicted_id = np.argmax(last_token_logits)

                try:
                    token = self.processor.tokenizer.decode([predicted_id])
                    print(f"Предсказанный токен: '{token}' (ID: {predicted_id})")
                except Exception as decode_error:
                    print(
                        f"Не удалось декодировать токен ID {predicted_id}: {decode_error}"
                    )

            return ort_outputs[0]

        except Exception as e:
            print(f"❌ ONNX инференс не работает: {e}")
            print(f"Детали ошибки: {str(e)}")
            return None

    def benchmark_performance(self, num_runs: int = 100):
        """Бенчмарк производительности ONNX модели"""
        import time

        if self.session is None:
            raise ValueError("ONNX модель не загружена.")

        dummy_image = np.random.randn(1, 3, 384, 384).astype(np.float32)

        token_id = getattr(self.processor.tokenizer, "bos_token_id", None)
        if token_id is None:
            token_id = getattr(self.processor.tokenizer, "cls_token_id", 101)

        dummy_input_ids = np.array([[token_id] * 16], dtype=np.int64)

        ort_inputs = {"image": dummy_image, "input_ids": dummy_input_ids}

        print(f"Запуск бенчмарка ({num_runs} итераций)...")

        for _ in range(10):
            self.session.run(None, ort_inputs)

        latencies = []
        for i in range(num_runs):
            start_time = time.time()
            self.session.run(None, ort_inputs)
            end_time = time.time()

            latency = (end_time - start_time) * 1000
            latencies.append(latency)

            if (i + 1) % 20 == 0:
                print(f"Завершено {i + 1}/{num_runs} итераций")

        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print("\nСтатистика производительности ONNX модели:")
        print(f"Среднее время инференса: {avg_latency:.2f} мс")
        print(f"P50 latency: {p50_latency:.2f} мс")
        print(f"P95 latency: {p95_latency:.2f} мс")
        print(f"P99 latency: {p99_latency:.2f} мс")

        return {
            "avg_latency": avg_latency,
            "p50_latency": p50_latency,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency,
            "latencies": latencies,
        }


def main():
    """Основная функция для тестирования ONNX модели"""
    onnx_path = "models/blip_model.onnx"

    tester = ONNXModelTester(onnx_path)

    tester.load_onnx_model()

    tester.test_inference()

    tester.benchmark_performance(num_runs=50)

    print("\nТестирование ONNX модели завершено!")


if __name__ == "__main__":
    main()
