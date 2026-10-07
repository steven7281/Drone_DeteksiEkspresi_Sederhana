import os
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

# add ground-control-station/AI ke python path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "AI",
        )
    ),
)

from model_runner import SCRFDFaceDetector

# network configuration
gcs_ip = "0.0.0.0"
gcs_port = 5001

# video configuration
width = 640
height = 480
frame_size = width * height * 3

# model configuration
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

detector = SCRFDFaceDetector(model_path, input_size=224)

print("SCRFD detector berhasil dibuat")

# ffmpeg configuration
ffmpeg_command = [
    "ffmpeg",
    "-fflags",
    "nobuffer+discardcorrupt+genpts",
    "-flags",
    "low_delay",
    "-analyzeduration",
    "0",
    "-probesize",
    "32",
    "-max_delay",
    "0",
    "-i",
    f"udp://{gcs_ip}:{gcs_port}?fifo_size=1000000&overrun_nonfatal=1",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "bgr24",
    "pipe:1",
]

try:
    decoder = subprocess.Popen(
        ffmpeg_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8,
    )
except FileNotFoundError:
    print("ERROR: FFmpeg tidak ditemukan")
    print("Pastikan command 'ffmpeg' tersedia di terminal")
    sys.exit()

print(f"Listening H.264 UDP on {gcs_ip}:{gcs_port}")
print("Menunggu video dari drone...")
print("Tekan Q untuk keluar")

latest_frame = None
latest_frame_time = None

frame_lock = threading.Lock()
stop_event = threading.Event()

received_frames = 0
processed_frames = 0

received_frame_lock = threading.Lock()
processed_frame_lock = threading.Lock()

inference_times = deque(maxlen=100)


def receive_frames():
    global latest_frame
    global latest_frame_time
    global received_frames

    while not stop_event.is_set():
        raw_frame = decoder.stdout.read(frame_size)

        if len(raw_frame) != frame_size:
            print("Frame tidak lengkap atau stream berhenti")
            stop_event.set()
            break

        frame = np.frombuffer(
            raw_frame,
            dtype=np.uint8,
        ).reshape(
            (height, width, 3)
        ).copy()

        with frame_lock:
            latest_frame = frame
            latest_frame_time = time.perf_counter()

        with received_frame_lock:
            received_frames += 1


receiver_thread = threading.Thread(
    target=receive_frames,
    daemon=True,
)

receiver_thread.start()

last_report_time = time.perf_counter()
last_received_frames = 0
last_processed_frames = 0

try:
    while not stop_event.is_set():

        with frame_lock:
            if latest_frame is None:
                frame = None
                frame_time = None
            else:
                frame = latest_frame.copy()
                frame_time = latest_frame_time

        if frame is None:
            time.sleep(0.001)
            continue

        frame_age = (time.perf_counter() - frame_time) * 1000

        start_time = time.perf_counter()

        if processed_frames % 4 == 0:
            try:
                last_faces = detector.run(frame)
            except Exception as e:
                print(f"Deteksi gagal ({e})")
                last_faces = []
        faces = last_faces

        inference_time = (time.perf_counter() - start_time) * 1000
        inference_times.append(inference_time)

        with processed_frame_lock:
            processed_frames += 1

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"scrfd: {inference_time:.1f} ms",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"frame age: {frame_age:.1f} ms",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow(
            "GCS - H264 Drone Camera",
            frame,
        )

        current_time = time.perf_counter()

        if current_time - last_report_time >= 1.0:

            elapsed_time = current_time - last_report_time

            with received_frame_lock:
                current_received_frames = received_frames

            with processed_frame_lock:
                current_processed_frames = processed_frames

            received_fps = (
                current_received_frames - last_received_frames
            ) / elapsed_time

            processed_fps = (
                current_processed_frames - last_processed_frames
            ) / elapsed_time

            if len(inference_times) > 0:
                average_inference = sum(inference_times) / len(inference_times)
            else:
                average_inference = 0.0

            print(
                f"received fps: {received_fps:.1f} | "
                f"processed fps: {processed_fps:.1f} | "
                f"SCRFD: {average_inference:.1f} ms | "
                f"frame age: {frame_age:.1f} ms"
            )

            last_received_frames = current_received_frames
            last_processed_frames = current_processed_frames
            last_report_time = current_time

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("\nTest dihentikan")

finally:
    stop_event.set()

    decoder.terminate()
    decoder.wait()

    cv2.destroyAllWindows()

    print("GCS - H264 Drone Camera selesai")
