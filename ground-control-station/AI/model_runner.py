"""Loader + runner model AI untuk GCS.

Model diambil dari folder:
    AI/model/<namaModel>/

h264_receiver.py hanya memanggil load_model() dan run() dari modul ini —
logika pemanggilan model tidak ditulis di script receiver.
"""

import os

import cv2
import numpy as np

AI_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(AI_DIR, "model")
MODEL_NAME = "mobilenetv3_small"
MODEL_DIR = os.path.join(MODEL_ROOT, MODEL_NAME)

MODEL_EXTENSIONS = [".tflite", ".onnx", ".h5", ".keras", ".pt", ".pth"]


def find_model_file():
    if not os.path.isdir(MODEL_DIR):
        return None
    for name in sorted(os.listdir(MODEL_DIR)):
        if any(name.lower().endswith(ext) for ext in MODEL_EXTENSIONS):
            return os.path.join(MODEL_DIR, name)
    return None


def _preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    # setara dengan tf.keras.applications.mobilenet_v3.preprocess_input
    img = img / 127.5 - 1.0
    return img


class ModelRunner:
    def __init__(self, model_path):
        self.model_path = model_path
        self.kind = None
        self.model = None
        self._load()

    def _load(self):
        ext = os.path.splitext(self.model_path)[1].lower()

        if ext == ".tflite":
            import tensorflow as tf

            interpreter = tf.lite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            self.kind, self.model = "tflite", interpreter

        elif ext == ".onnx":
            self.kind, self.model = "onnx", cv2.dnn.readNetFromONNX(self.model_path)

        elif ext in (".h5", ".keras"):
            import tensorflow as tf

            self.kind, self.model = "keras", tf.keras.models.load_model(self.model_path)

        elif ext in (".pt", ".pth"):
            import torch
            from torchvision import models

            model = models.mobilenet_v3_small(weights=None)
            state = torch.load(self.model_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            model.eval()
            self.kind, self.model = "torch", model

    def run(self, frame):
        img = _preprocess(frame)

        if self.kind == "tflite":
            inp = self.model.get_input_details()
            out = self.model.get_output_details()
            self.model.set_tensor(inp[0]["index"], np.expand_dims(img.astype(inp[0]["dtype"]), 0))
            self.model.invoke()
            return self.model.get_tensor(out[0]["index"])

        if self.kind == "onnx":
            blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(224, 224), swapRB=False, crop=False)
            self.model.setInput(blob)
            return self.model.forward()

        if self.kind == "keras":
            return self.model.predict(np.expand_dims(img, 0), verbose=0)

        if self.kind == "torch":
            import torch

            tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                return self.model(tensor).numpy()

        return None


def load_model():
    path = find_model_file()
    if path is None:
        return None, None
    return ModelRunner(path), path


class FaceTracker:
    """Deteksi wajah dengan Haar Cascade bawaan OpenCV (offline, tanpa download)."""

    def __init__(self):
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Gagal memuat cascade: {cascade_path}")

    def run(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
