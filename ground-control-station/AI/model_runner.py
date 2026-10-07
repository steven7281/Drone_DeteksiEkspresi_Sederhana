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


class SCRFDFaceDetector:
    """Deteksi wajah dengan SCRFD (det_10g.onnx) via OpenCV DNN.

    Menggantikan FaceTracker/Haar — lebih akurat untuk wajah kecil/miring.
    """

    def __init__(self, model_path, input_size=640, conf_thres=0.5, nms_thres=0.4):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self._center_cache = {}

    def _anchor_centers(self, height, width, stride):
        key = (height, width, stride)
        if key in self._center_cache:
            return self._center_cache[key]
        y, x = np.mgrid[0:height, 0:width]
        centers = np.stack([x, y], axis=-1).astype(np.float32)
        centers = (centers * stride).reshape(-1, 2)
        centers = np.repeat(centers, 2, axis=0)  # 2 anchor per lokasi
        self._center_cache[key] = centers
        return centers

    def run(self, frame):
        h0, w0 = frame.shape[:2]
        scale = self.input_size / max(h0, w0)
        h1, w1 = int(h0 * scale), int(w0 * scale)
        resized = cv2.resize(frame, (w1, h1))

        det_img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        det_img[:h1, :w1] = resized

        blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128.0, (self.input_size, self.input_size),
            (127.5, 127.5, 127.5), swapRB=False,
        )
        outputs = self.session.run(None, {self.input_name: blob})

        # outputs dikelompokkan per stride: (scores, bboxes, kps)
        strides = [8, 16, 32]
        all_boxes, all_scores = [], []

        for i, stride in enumerate(strides):
            scores = outputs[i]                # (N,1)
            bboxes = outputs[i + 3]            # (N,4)
            feat_h = self.input_size // stride
            feat_w = self.input_size // stride
            centers = self._anchor_centers(feat_h, feat_w, stride)

            scores = scores.reshape(-1)
            keep = np.where(scores > self.conf_thres)[0]
            if len(keep) == 0:
                continue

            b = bboxes[keep] * stride
            c = centers[keep]
            x1 = c[:, 0] - b[:, 0]
            y1 = c[:, 1] - b[:, 1]
            x2 = c[:, 0] + b[:, 2]
            y2 = c[:, 1] + b[:, 3]

            all_boxes.append(np.stack([x1, y1, x2 - x1, y2 - y1], axis=1))
            all_scores.append(scores[keep])

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes) / scale
        scores = np.concatenate(all_scores)

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self.conf_thres, self.nms_thres
        )
        faces = []
        for idx in np.array(indices).flatten():
            x, y, w, h = boxes[idx].astype(int)
            x, y = max(0, x), max(0, y)
            w, h = min(w, w0 - x), min(h, h0 - y)
            if w > 0 and h > 0:
                faces.append((x, y, w, h))
        return faces
