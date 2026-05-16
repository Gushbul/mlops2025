from src.batch_optimizer import BatchOptimizer
import os


def main():
    """
    Демонстрация подбора оптимального размера батча
    """
    print("=== Шаг 3: Оптимизация размера батча для ONNX модели ===\n")

    onnx_path = "models/blip_model.onnx"
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX модель не найдена: {onnx_path}")
        print("\nДля продолжения скопируйте модель из step1:")
        print("cp ../step1_onnx_model/models/blip_model.onnx models/")
        return

    print(f"✅ Найдена ONNX модель: {onnx_path}")

    optimizer = BatchOptimizer(onnx_path)
    optimizer.load_model()

    print("\n1. Быстрый тест (мало итераций)")
    optimal_batch_quick, results_quick = optimizer.find_optimal_batch_size(
        max_batch_size=4, num_iterations=20, max_memory_mb=300, target_p95_ms=50
    )

    print(f"\nБыстрый тест - оптимальный batch_size: {optimal_batch_quick}")

    print("\n" + "=" * 50)
    print("2. Полный тест (больше итераций и размеров)")

    optimal_batch_full, results_full = optimizer.find_optimal_batch_size(
        max_batch_size=8, num_iterations=40, max_memory_mb=500, target_p95_ms=100
    )

    results_full.to_csv("results/optimization_results.csv", index=False)
    print("\n📊 Результаты сохранены: results/optimization_results.csv")

    optimizer.plot_results(results_full, "results/batch_optimization.png")

    print("\n" + "=" * 50)
    print("🎯 ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
    print("=" * 50)
    print(f"Быстрый тест - оптимальный batch_size: {optimal_batch_quick}")
    print(f"Полный тест - оптимальный batch_size: {optimal_batch_full}")

    print("\nТоп-3 размера батча по p95 latency per sample:")
    top_batches = results_full.nsmallest(3, "p95_latency_per_sample_ms")
    for i, (_, row) in enumerate(top_batches.iterrows(), 1):
        print(
            f"{i}. batch_size={int(row['batch_size'])}: "
            f"p95={row['p95_latency_per_sample_ms']:.2f}ms/sample, "
            f"throughput={row['throughput_samples_per_sec']:.1f} samples/sec"
        )

    print("\n✅ Оптимизация завершена!")
    print("📈 Смотрите визуализацию: results/batch_optimization.png")


if __name__ == "__main__":
    main()
