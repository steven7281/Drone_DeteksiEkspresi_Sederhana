import os
import statistics
import sys
import time

import cv2

# add folder AI ke python path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AI")),
)

from model_runner import SCRFDFaceDetector

# konfigurasi model
model_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "AI",
        "model",
        "scrfd_2.5g_bnkps",
        "scrfd_2.5g_kps.onnx",
    )
)

# konfigurasi benchmark
input_sizes = [640, 480, 320]
warmup_iterations = 10
benchmark_iterations = 100

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Gagal membuka camera")
    sys.exit()

print("Camera berhasil dibuka")

ret, frame = camera.read()

if not ret:
    print("Gagal read frame")
    camera.release()
    sys.exit()

height, width = frame.shape[:2]
print(f"Resolution: {width}x{height}")
print()

for input_size in input_sizes:
    print(f"Benchmark SCRFD {input_size}x{input_size}")

    detector = SCRFDFaceDetector(model_path, input_size=input_size)

    # warmup
    for _ in range(warmup_iterations):
        detector.run(frame)

    inference_times = []

    for _ in range(benchmark_iterations):
        start_time = time.perf_counter()
        detector.run(frame)
        elapsed_time = time.perf_counter() - start_time
        inference_times.append(elapsed_time * 1000)

    average_time = statistics.mean(inference_times)
    median_time = statistics.median(inference_times)
    sorted_times = sorted(inference_times)
    p95_index = int(0.95 * len(sorted_times)) - 1
    p95_time = sorted_times[p95_index]
    max_time = max(inference_times)
    estimated_fps = 1000 / average_time

    print(f"Average: {average_time:.2f} ms")
    print(f"Median: {median_time:.2f} ms")
    print(f"P95: {p95_time:.2f} ms")
    print(f"Max: {max_time:.2f} ms")
    print(f"Estimated FPS: {estimated_fps:.2f}")
    print()

camera.release()
print("SCRFD benchmark selesai")
