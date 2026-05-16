import requests
import time


def test_onnx_api():
    """
    Тестирование FastAPI с ONNX моделью и реальным изображением
    """
    base_url = "http://localhost:8000"

    print("🧪 Тестирование ONNX FastAPI с реальным изображением\n")

    print("1. Health Check:")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health OK")
            print(f"   Model: {health_data.get('model_name', 'unknown')}")
            print(f"   Type: ONNX - {health_data.get('onnx_path', 'unknown')}")
        else:
            print(f"❌ Health failed: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Не могу подключиться к серверу: {e}")
        print("Убедитесь что FastAPI сервис запущен: python main.py")
        return

    print("\n2. Тест инференса с изображением:")
    image_path = "test_images/img.jpg"

    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/jpeg")}

            print(f"Отправляем изображение: {image_path}")
            start_time = time.time()

            response = requests.post(f"{base_url}/predict", files=files, timeout=60)

            end_time = time.time()
            request_time = (end_time - start_time) * 1000

            print(f"HTTP Response: {response.status_code}")
            print(f"Request time: {request_time:.2f} ms")

            if response.status_code == 200:
                result = response.json()
                print("✅ ONNX Inference успешно!")
                print(f"   Prediction: {result['result']['prediction']}")
                print(f"   Model type: {result['result']['model_type']}")
                print(f"   Success: {result['result']['success']}")
                print(
                    f"   Inference time: {result['result']['timing']['inference_ms']:.2f} ms"
                )

                if "onnx_details" in result["result"]:
                    onnx_info = result["result"]["onnx_details"]
                    print(
                        f"   ONNX token ID: {onnx_info.get('predicted_token_id', 'N/A')}"
                    )
                    print(
                        f"   ONNX logits shape: {onnx_info.get('logits_shape', 'N/A')}"
                    )

            else:
                print(f"❌ Inference failed: {response.text}")

    except FileNotFoundError:
        print(f"❌ Изображение не найдено: {image_path}")
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")

    print("\n✅ Тестирование ONNX API завершено!")


if __name__ == "__main__":
    test_onnx_api()
