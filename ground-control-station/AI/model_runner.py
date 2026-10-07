"""Loader + runner model AI untuk GCS (full GPU via ONNX Runtime).

- SCRFD (deteksi wajah) via onnxruntime CUDA
- MobileNetV3 small (.onnx) via onnxruntime CUDA
"""

import os

import cv2
import numpy as np

# pastikan DLL CUDA/cuDNN dari paket pip nvidia-* bisa ditemukan
_NV_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(cv2.__file__))),
    "nvidia",
)
for _sub in ["cudnn", "cublas", "cuda_runtime", "cufft", "nvjitlink"]:
    _bin = os.path.join(_NV_DIR, _sub, "bin")
    if os.path.isdir(_bin) and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(_bin)
        except OSError:
            pass
    if os.path.isdir(_bin) and _bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")

AI_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(AI_DIR, "model")
MODEL_NAME = "mobilenetv3_small"
MODEL_DIR = os.path.join(MODEL_ROOT, MODEL_NAME)

MODEL_EXTENSIONS = [".onnx"]


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
        import onnxruntime as ort

        self.model = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        print("MobileNetV3 providers:", self.model.get_providers())

    def run(self, frame):
        img = _preprocess(frame)
        inp = np.expand_dims(img, 0).astype(np.float32)
        return self.model.run(None, {self.model.get_inputs()[0].name: inp})


def load_model():
    path = find_model_file()
    if path is None:
        return None, None
    return ModelRunner(path), path


class SCRFDFaceDetector:
    """Deteksi wajah dengan SCRFD (.onnx) via onnxruntime CUDA."""

    def __init__(self, model_path, input_size=640, conf_thres=0.5, nms_thres=0.4, providers=None):
        import onnxruntime as ort

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
        )
        print("SCRFD providers:", self.session.get_providers())
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
