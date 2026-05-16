import time
import numpy as np
import onnxruntime as ort
from transformers import BlipProcessor
import psutil
import pandas as pd
from typing import Dict, Tuple, Optional
import os
import matplotlib.pyplot as plt


class BatchOptimizer:
    """
    Оптимизатор размера батча для ONNX модели на основе p95 latency
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
        """Загрузка ONNX модели"""
        if self.loaded:
            return

        print(f"Загрузка ONNX модели: {self.onnx_path}")

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        self.processor = BlipProcessor.from_pretrained(self.model_name)

        self.loaded = True
        print("Модель загружена успешно")

    def prepare_dummy_batch(self, batch_size: int) -> dict:
        """Подготовка dummy батча для тестирования"""

        dummy_images = np.random.randn(batch_size, 3, 384, 384).astype(np.float32)

        dummy_input_ids = np.array(
            [[30522] * 16 for _ in range(batch_size)], dtype=np.int64
        )

        return {"image": dummy_images, "input_ids": dummy_input_ids}

    def benchmark_batch_size(self, batch_size: int, num_iterations: int = 50) -> Dict:
        """
        Бенчмарк для конкретного размера батча

        Args:
            batch_size: Размер батча для тестирования
            num_iterations: Количество итераций для статистики

        Returns:
            Словарь с метриками производительности
        """
        if not self.loaded:
            raise ValueError("Модель не загружена")

        print(f"Тестирование batch_size={batch_size}, итераций={num_iterations}")

        batch_data = self.prepare_dummy_batch(batch_size)

        for _ in range(5):
            try:
                self.session.run(None, batch_data)
            except Exception as e:
                return {"batch_size": batch_size, "error": str(e), "success": False}

        latencies = []
        memory_usage = []

        for i in range(num_iterations):

            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024

            start_time = time.time()
            try:
                self.session.run(None, batch_data)
                end_time = time.time()

                latency = (end_time - start_time) * 1000
                latencies.append(latency)

                memory_after = process.memory_info().rss / 1024 / 1024
                memory_usage.append(memory_after - memory_before)

            except Exception as e:
                return {"batch_size": batch_size, "error": str(e), "success": False}

            if (i + 1) % 10 == 0:
                print(f"  Завершено {i + 1}/{num_iterations} итераций")

        latencies = np.array(latencies)
        memory_usage = np.array(memory_usage)

        latencies_per_sample = latencies / batch_size

        results = {
            "batch_size": batch_size,
            "success": True,
            "total_latency": {
                "mean_ms": np.mean(latencies),
                "p50_ms": np.percentile(latencies, 50),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99),
                "std_ms": np.std(latencies),
            },
            "per_sample_latency": {
                "mean_ms": np.mean(latencies_per_sample),
                "p50_ms": np.percentile(latencies_per_sample, 50),
                "p95_ms": np.percentile(latencies_per_sample, 95),
                "p99_ms": np.percentile(latencies_per_sample, 99),
                "std_ms": np.std(latencies_per_sample),
            },
            "throughput": {
                "samples_per_second": batch_size * 1000 / np.mean(latencies)
            },
            "memory": {
                "mean_mb": np.mean(memory_usage),
                "max_mb": np.max(memory_usage),
                "std_mb": np.std(memory_usage),
            },
            "raw_data": {
                "latencies_ms": latencies.tolist(),
                "memory_usage_mb": memory_usage.tolist(),
            },
        }

        return results

    def find_optimal_batch_size(
        self,
        max_batch_size: int = 16,
        num_iterations: int = 50,
        max_memory_mb: Optional[int] = None,
        target_p95_ms: Optional[float] = None,
    ) -> Tuple[int, pd.DataFrame]:
        """
        Поиск оптимального размера батча

        Args:
            max_batch_size: Максимальный размер батча для тестирования
            num_iterations: Количество итераций для каждого размера
            max_memory_mb: Ограничение по памяти (MB)
            target_p95_ms: Целевое значение p95 latency per sample

        Returns:
            Оптимальный размер батча и DataFrame с результатами
        """
        print("=== Поиск оптимального размера батча ===\n")

        batch_sizes = [2**i for i in range(int(np.log2(max_batch_size)) + 1)]
        batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]
        batch_sizes = [1] + batch_sizes

        print(f"Тестируемые размеры батчей: {batch_sizes}")
        print(f"Итераций для каждого размера: {num_iterations}")

        results = []

        for batch_size in batch_sizes:
            result = self.benchmark_batch_size(batch_size, num_iterations)

            if result["success"]:
                results.append(
                    {
                        "batch_size": batch_size,
                        "p95_latency_total_ms": result["total_latency"]["p95_ms"],
                        "p95_latency_per_sample_ms": result["per_sample_latency"][
                            "p95_ms"
                        ],
                        "mean_latency_per_sample_ms": result["per_sample_latency"][
                            "mean_ms"
                        ],
                        "throughput_samples_per_sec": result["throughput"][
                            "samples_per_second"
                        ],
                        "memory_max_mb": result["memory"]["max_mb"],
                        "memory_mean_mb": result["memory"]["mean_mb"],
                    }
                )
                print(
                    f"✅ batch_size={batch_size}: p95={result['per_sample_latency']['p95_ms']:.2f}ms/sample, "
                    f"throughput={result['throughput']['samples_per_second']:.2f} samples/sec"
                )
            else:
                print(
                    f"❌ batch_size={batch_size}: {result.get('error', 'Unknown error')}"
                )

        if not results:
            raise ValueError("Ни один размер батча не прошел тестирование")

        df = pd.DataFrame(results)

        valid_df = df.copy()

        if max_memory_mb:
            valid_df = valid_df[valid_df["memory_max_mb"] <= max_memory_mb]
            print(f"\nПрименено ограничение по памяти: <= {max_memory_mb} MB")

        if target_p95_ms:
            valid_df = valid_df[valid_df["p95_latency_per_sample_ms"] <= target_p95_ms]
            print(f"Применено ограничение по p95 latency: <= {target_p95_ms} ms/sample")

        if valid_df.empty:
            print("⚠️ Ни один размер батча не удовлетворяет ограничениям")
            print("Выбираем лучший по p95 latency per sample из всех вариантов")
            valid_df = df

        optimal_idx = valid_df["p95_latency_per_sample_ms"].idxmin()
        optimal_batch_size = valid_df.loc[optimal_idx, "batch_size"]

        print(f"\n🎯 Оптимальный размер батча: {optimal_batch_size}")
        print(
            f"   p95 latency per sample: {valid_df.loc[optimal_idx, 'p95_latency_per_sample_ms']:.2f} ms"
        )
        print(
            f"   Throughput: {valid_df.loc[optimal_idx, 'throughput_samples_per_sec']:.2f} samples/sec"
        )
        print(f"   Memory usage: {valid_df.loc[optimal_idx, 'memory_max_mb']:.1f} MB")

        return optimal_batch_size, df

    def plot_results(
        self, df: pd.DataFrame, save_path: str = "results/batch_optimization.png"
    ):
        """Визуализация результатов оптимизации"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Batch Size Optimization Results", fontsize=16)

        axes[0, 0].plot(df["batch_size"], df["p95_latency_per_sample_ms"], "bo-")
        axes[0, 0].set_xlabel("Batch Size")
        axes[0, 0].set_ylabel("P95 Latency per Sample (ms)")
        axes[0, 0].set_title("P95 Latency per Sample vs Batch Size")
        axes[0, 0].grid(True)

        axes[0, 1].plot(df["batch_size"], df["throughput_samples_per_sec"], "go-")
        axes[0, 1].set_xlabel("Batch Size")
        axes[0, 1].set_ylabel("Throughput (samples/sec)")
        axes[0, 1].set_title("Throughput vs Batch Size")
        axes[0, 1].grid(True)

        axes[1, 0].plot(df["batch_size"], df["memory_max_mb"], "ro-")
        axes[1, 0].set_xlabel("Batch Size")
        axes[1, 0].set_ylabel("Max Memory Usage (MB)")
        axes[1, 0].set_title("Memory Usage vs Batch Size")
        axes[1, 0].grid(True)

        efficiency = df["throughput_samples_per_sec"] / df["memory_max_mb"]
        axes[1, 1].plot(df["batch_size"], efficiency, "mo-")
        axes[1, 1].set_xlabel("Batch Size")
        axes[1, 1].set_ylabel("Efficiency (samples/sec per MB)")
        axes[1, 1].set_title("Efficiency vs Batch Size")
        axes[1, 1].grid(True)

        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"График сохранен: {save_path}")
        plt.close()


def main():
    """Демонстрация оптимизации размера батча"""
    onnx_path = "models/blip_model.onnx"

    if not os.path.exists(onnx_path):
        print(f"❌ ONNX модель не найдена: {onnx_path}")
        print(
            "Скопируйте модель из step1: cp ../step1_onnx_model/models/blip_model.onnx models/"
        )
        return

    optimizer = BatchOptimizer(onnx_path)
    optimizer.load_model()

    optimal_batch_size, results_df = optimizer.find_optimal_batch_size(
        max_batch_size=8,
        num_iterations=30,
        max_memory_mb=500,
        target_p95_ms=100,
    )

    results_df.to_csv("results/optimization_results.csv", index=False)
    print("Результаты сохранены: results/optimization_results.csv")

    optimizer.plot_results(results_df)

    print(f"\n✅ Оптимизация завершена. Рекомендуемый batch_size: {optimal_batch_size}")


if __name__ == "__main__":
    main()
