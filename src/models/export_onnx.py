"""Export du meilleur YOLO11 en ONNX : FP32, FP16 et INT8 (quantification statique).

Lancement :
    python -m src.models.export_onnx

FP32  : modele de reference, 4 octets par poids.
FP16  : 2 octets par poids, utile surtout sur GPU.
INT8  : 1 octet par poids et activations quantifiees, calibre sur des images
        d'entrainement (jamais de test : pas de fuite).
"""

import importlib
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from src.utils.mlflow_utils import load_params

BEST_PT = Path("models/yolo11_best.pt")
ONNX_DIR = Path("models/onnx")
CALIB_DIR = Path("data/subset/images/train")
HEAD_RE = re.compile(r"/model\.(\d+)/")


def backend_inference():
    """Reutilise exactement le pre-traitement de l'API (une seule source)."""
    backend = str(Path(__file__).resolve().parents[2] / "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)
    return importlib.import_module("app.inference")


class CalibrationReader:
    """Fournit des images au quantificateur pour mesurer les plages d'activation."""

    def __init__(self, input_name, imgsz, count, seed=0):
        from PIL import Image

        inference = backend_inference()
        paths = sorted(CALIB_DIR.glob("*.jpg"))
        random.Random(seed).shuffle(paths)
        self.batches = []
        for path in paths[:count]:
            image = np.asarray(Image.open(path).convert("RGB"))
            square, _, _ = inference.letterbox(image, imgsz)
            self.batches.append({input_name: inference.to_tensor(square)})
        self.index = 0

    def get_next(self):
        if self.index >= len(self.batches):
            return None
        self.index += 1
        return self.batches[self.index - 1]

    def rewind(self):
        self.index = 0


def export_float(imgsz, half):
    import torch
    from ultralytics import YOLO

    if half and not torch.cuda.is_available():
        print("FP16 ignore : l'export demi-precision demande un GPU.")
        return None
    path = YOLO(str(BEST_PT)).export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        device=0 if half else "cpu",
        simplify=True,
        dynamic=False,
    )
    target = ONNX_DIR / f"touchfree_{'fp16' if half else 'fp32'}.onnx"
    shutil.move(path, target)
    return target


def head_nodes_to_exclude(onnx_path):
    """Noeuds de la tete Detect a garder en flottant (sauf ses convolutions).

    La tete concatene les coordonnees (0 a 512 px) et les scores (0 a 1) dans un
    meme tenseur : avec une seule echelle INT8, les scores seraient ecrases.
    """
    import onnx

    graph = onnx.load(str(onnx_path)).graph
    indices = [int(m.group(1)) for n in graph.node if (m := HEAD_RE.search(n.name))]
    if not indices:
        return []
    head = f"/model.{max(indices)}/"
    return [
        n.name
        for n in graph.node
        if head in n.name and (n.op_type != "Conv" or "/dfl/" in n.name)
    ]


def export_int8(fp32_path, imgsz, calib_count):
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prepared = ONNX_DIR / "touchfree_fp32_prep.onnx"
    quant_pre_process(str(fp32_path), str(prepared))
    input_name = ort.InferenceSession(str(prepared)).get_inputs()[0].name
    target = ONNX_DIR / "touchfree_int8.onnx"
    quantize_static(
        str(prepared),
        str(target),
        CalibrationReader(input_name, imgsz, calib_count),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=head_nodes_to_exclude(prepared),
    )
    prepared.unlink()
    return target


def main():
    params = load_params()
    cfg = params["export"]
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    fp32 = export_float(cfg["imgsz"], half=False)
    export_float(cfg["imgsz"], half=True)
    export_int8(fp32, cfg["imgsz"], cfg["calib_images"])
    for path in sorted(ONNX_DIR.glob("*.onnx")):
        print(f"{path.name:28s} {path.stat().st_size / 1e6:6.1f} Mo")


if __name__ == "__main__":
    main()
