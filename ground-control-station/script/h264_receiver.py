import os
import subprocess
import sys
import time

import cv2
import numpy as np

HOST = "0.0.0.0"
PORT = 5001

WIDTH = 640
HEIGHT = 480

FRAME_SIZE = WIDTH * HEIGHT * 3

_AI_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AI")
)
sys.path.insert(0, _AI_DIR)

from model_runner import FaceTracker, load_model

MODEL_DIR = os.path.normpath(
    os.path.join(_AI_DIR, "model", "mobilenetv3_small")
)


def install_model():
    """Unduh & ekspor MobileNetV3 small ke AI/model/mobilenetv3_small/."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_DIR, "mobilenetv3_small.tflite")

    if os.path.exists(out_path):
        return out_path

    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: tensorflow belum terinstal. Jalankan: pip install tensorflow")
        return None

    print("Mengunduh & mengekspor model MobileNetV3 small...")
    model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=True, weights="imagenet"
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    with open(out_path, "wb") as f:
        f.write(converter.convert())
    print(f"Model disimpan: {out_path}")
    return out_path


ffmpeg_command = [
    "ffmpeg",
    "-fflags",
    "nobuffer",
    "-flags",
    "low_delay",
    "-i",
    f"udp://{HOST}:{PORT}?fifo_size=1000000&overrun_nonfatal=1",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "bgr24",
    "pipe:1",
]

print(f"Listening H.264 UDP on {HOST}:{PORT}")

try:
    decoder = subprocess.Popen(
        ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8
    )
except FileNotFoundError:
    print("ERROR: FFmpeg tidak ditemukan.")
    sys.exit(1)

cv2.namedWindow("GCS - H264 Drone Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("GCS - H264 Drone Camera", WIDTH, HEIGHT)

detector, model_path = load_model()
if detector is None:
    print("Model belum ada, menjalankan proses install model...")
    install_model()
    detector, model_path = load_model()

if detector is None:
    print("WARN: Model tidak ditemukan di AI/model/mobilenetv3_small/. Inferensi di-skip.")
else:
    print(f"Model dimuat: {model_path}")

face_tracker = FaceTracker()
print("Face tracker siap (Haar Cascade).")

try:
    while True:
        raw_frame = decoder.stdout.read(FRAME_SIZE)

        if len(raw_frame) != FRAME_SIZE:
            print("Frame tidak lengkap atau stream berhenti.")
            break

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()

        t0 = time.perf_counter()
        faces = face_tracker.run(frame)
        t1 = time.perf_counter()
        face_ms = (t1 - t0) * 1000.0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(
            frame,
            f"Face: {len(faces)} | {face_ms:.1f} ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        print(f"Face detection: {len(faces)} wajah | {face_ms:.1f} ms")

        if detector is not None:
            t0 = time.perf_counter()
            detector.run(frame)
            t1 = time.perf_counter()
            infer_ms = (t1 - t0) * 1000.0

            cv2.putText(
                frame,
                f"Inference: {infer_ms:.1f} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            print(f"Inference: {infer_ms:.1f} ms")

        cv2.imshow("GCS - H264 Drone Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("\nReceiver dihentikan.")

finally:
    decoder.terminate()
    decoder.wait()

    cv2.destroyAllWindows()

    print("H.264 receiver berhenti.")
