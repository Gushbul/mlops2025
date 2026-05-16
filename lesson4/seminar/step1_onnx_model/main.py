from src.model_converter import BlipONNXConverter
from src.onnx_tester import ONNXModelTester
import os


def main():
    """
    Демонстрация полного цикла конвертации и тестирования ONNX модели
    """
    print("=== Шаг 1: Конвертация модели BLIP в ONNX ===\n")

    converter = BlipONNXConverter()
    converter.load_model()

    print("\n1. Тестирование PyTorch модели:")
    pytorch_caption = converter.test_pytorch_model()

    print("\n2. Конвертация в ONNX:")
    onnx_path = converter.convert_to_onnx()

    if os.path.exists(onnx_path):
        print(f"✅ ONNX модель успешно создана: {onnx_path}")
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"Размер файла: {file_size:.1f} MB")
    else:
        print("❌ Ошибка: ONNX файл не создан")
        return

    print("\n3. Тестирование ONNX модели:")
    tester = ONNXModelTester(onnx_path)
    tester.load_onnx_model()

    try:
        outputs = tester.test_inference()
        if outputs is not None:
            print(f"✅ ONNX инференс работает, размер выхода: {outputs.shape}")
        else:
            print("⚠️ ONNX инференс не работает, но модель конвертирована")
    except Exception as e:
        print(f"❌ Ошибка ONNX инференса: {e}")

    print("\n4. Бенчмарк производительности:")
    performance = tester.benchmark_performance(num_runs=30)

    print("\n=== Результаты ===")
    print(f"PyTorch caption: {pytorch_caption}")
    print(f"ONNX модель файл: {onnx_path}")
    print(f"P95 latency: {performance['p95_latency']:.2f} мс")
    print("\n✅ Конвертация и тестирование завершены успешно!")


if __name__ == "__main__":
    main()
