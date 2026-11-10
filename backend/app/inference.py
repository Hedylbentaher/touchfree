"""Inference ONNX de YOLO11 : pre-traitement, decodage et NMS, en NumPy pur.

Ce module ne depend ni de PyTorch ni d'Ultralytics : l'image Docker reste legere.
Il est aussi reutilise par l'export INT8 (calibration) et le benchmark.
"""

import time

import numpy as np
from PIL import Image

PAD_VALUE = 114  # meme gris de remplissage qu'Ultralytics


def letterbox(image, size):
    """Redimensionne en gardant les proportions, puis complete en carre.

    Retourne l'image carree (uint8, HxWx3), le facteur d'echelle et le
    decalage (dx, dy) ajoute a gauche et en haut.
    """
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    resized = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    dx, dy = (size - new_w) // 2, (size - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = np.asarray(resized)
    return canvas, ratio, (dx, dy)


def to_tensor(square_image, dtype=np.float32):
    """HxWx3 uint8 -> 1x3xHxW normalise entre 0 et 1."""
    tensor = square_image.astype(np.float32) / 255.0
    return tensor.transpose(2, 0, 1)[None].astype(dtype)


def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression independante de la classe (class-agnostic).

    Une main = une seule boite : si deux boites se chevauchent trop,
    on garde la plus sure, meme si leurs classes different.
    """
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        rest = order[1:]
        x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / (areas[best] + areas[rest] - inter + 1e-9)
        order = rest[iou < iou_threshold]
    return keep


def decode(output, conf_threshold, iou_threshold, ratio, pad, orig_size):
    """Sortie YOLO11 (1, 4 + nb_classes, N) -> liste de detections en pixels.

    Les 4 premieres lignes sont cx, cy, w, h dans l'image carree ;
    les suivantes sont les scores de chaque classe.
    """
    preds = np.asarray(output, dtype=np.float32)[0].T  # (N, 4 + nb_classes)
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)
    mask = scores >= conf_threshold
    if not mask.any():
        return []
    cxcywh, scores, class_ids = preds[mask, :4], scores[mask], class_ids[mask]

    boxes = np.empty_like(cxcywh)
    boxes[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
    boxes[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
    boxes[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
    boxes[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2

    keep = nms(boxes, scores, iou_threshold)

    # On annule le letterbox pour revenir aux pixels de l'image d'origine.
    width, height = orig_size
    dx, dy = pad
    boxes = boxes[keep]
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - dx) / ratio).clip(0, width)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - dy) / ratio).clip(0, height)
    return [
        {
            "class_id": int(class_ids[k]),
            "confidence": float(scores[k]),
            "box": [round(float(v), 1) for v in box],
        }
        for k, box in zip(keep, boxes)
    ]


class OnnxDetector:
    """Charge un modele ONNX une seule fois et detecte les mains d'une image."""

    def __init__(self, model_path, class_names, imgsz=512, conf=0.35, iou=0.5):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_dtype = np.float16 if "float16" in model_input.type else np.float32
        self.class_names = list(class_names)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def preprocess(self, image):
        square, ratio, pad = letterbox(image, self.imgsz)
        return to_tensor(square, self.input_dtype), ratio, pad

    def predict(self, image):
        """image : tableau RGB HxWx3 uint8. Retourne (detections, temps en ms)."""
        start = time.perf_counter()
        tensor, ratio, pad = self.preprocess(image)
        output = self.session.run(None, {self.input_name: tensor})[0]
        height, width = image.shape[:2]
        detections = decode(output, self.conf, self.iou, ratio, pad, (width, height))
        for det in detections:
            cid = det["class_id"]
            det["label"] = (
                self.class_names[cid] if cid < len(self.class_names) else str(cid)
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return detections, elapsed_ms
