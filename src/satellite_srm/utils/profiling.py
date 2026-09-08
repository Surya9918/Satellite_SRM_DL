"""Benchmark profiler measuring inference latency, throughput (pixels/sec), and memory consumption."""
import time
from dataclasses import dataclass
from typing import Dict, Any
from satellite_srm.device import get_device_info

@dataclass
class BenchmarkResult:
    device: str
    model_name: str
    patch_size: int
    batch_size: int
    latency_ms: float
    images_per_sec: float
    pixels_per_sec: float
    memory_info: Dict[str, Any]

class BenchmarkProfiler:
    """Profiles deep learning super-resolution throughput."""

    @staticmethod
    def profile_inference(model, input_tensor, iterations: int = 5) -> BenchmarkResult:
        device_info = get_device_info()

        # Warmup
        for _ in range(2):
            _ = model(input_tensor)

        t0 = time.time()
        for _ in range(iterations):
            _ = model(input_tensor)
        total_time = time.time() - t0

        avg_latency = (total_time / iterations) * 1000.0  # ms
        batch_size = input_tensor.shape[0]
        h, w = input_tensor.shape[-2:]
        total_pixels = batch_size * h * w * iterations

        return BenchmarkResult(
            device=device_info["current_device"],
            model_name=model.__class__.__name__,
            patch_size=h,
            batch_size=batch_size,
            latency_ms=round(avg_latency, 2),
            images_per_sec=round((batch_size * iterations) / total_time, 2),
            pixels_per_sec=round(total_pixels / total_time, 2),
            memory_info=device_info
        )
