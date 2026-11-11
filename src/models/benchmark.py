"""Compare les variantes ONNX (precision, taille, latence CPU) et les 2 modeles.

Lancement :
    python -m src.models.benchmark
Produit reports/benchmark.json et reports/comparison.md, et les envoie a MLflow.
"""

import json
import time
from pathlib import Path

import numpy as np

from src.models.export_onnx import ONNX_DIR, backend_inference
from src.utils.mlflow_utils import load_params, setup_mlflow

DATA_YAML = Path("data/subset/data.yaml")
TEST_DIR = Path("data/subset/images/test")
REPORTS_DIR = Path("reports")


def accuracy(onnx_path, imgsz):
    """mAP sur le split test, calculee par Ultralytics directement sur l'ONNX."""
    from ultralytics import YOLO

    metrics = YOLO(str(onnx_path), task="detect").val(
        data=str(DATA_YAML),
        split="test",
        imgsz=imgsz,
        batch=1,
        device="cpu",
        plots=False,
        verbose=False,
    )
    return float(metrics.box.map), float(metrics.box.map50)


def latency(onnx_path, names, imgsz, runs):
    """Latence CPU de bout en bout (pre-traitement + modele + NMS), comme l'API."""
    from PIL import Image

    inference = backend_inference()
    detector = inference.OnnxDetector(onnx_path, names, imgsz=imgsz)
    paths = sorted(TEST_DIR.glob("*.jpg"))[:runs]
    images = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    for image in images[:10]:  # echauffement
        detector.predict(image)
    times = []
    for image in images:
        start = time.perf_counter()
        detector.predict(image)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def markdown_table(results, yolo, frcnn):
    lines = [
        "| Modele | mAP50-95 test | mAP50 test | Taille (Mo) | Latence CPU (ms) |",
        "|---|---|---|---|---|",
    ]
    if frcnn:
        lines.append(
            f"| Faster R-CNN MobileNetV3 (baseline, PyTorch) | {frcnn['map50_95']:.3f} "
            f"| {frcnn['map50']:.3f} | - | {frcnn['latency_ms']:.1f} "
            f"({frcnn['device']}) |"
        )
    if yolo:
        lines.append(
            f"| YOLO11n (PyTorch) | {yolo['map50_95']:.3f} | {yolo['map50']:.3f} "
            f"| - | {yolo['speed_ms']['inference']:.1f} (GPU) |"
        )
    for variant, r in results.items():
        if "error" in r:
            lines.append(f"| YOLO11n ONNX {variant} | erreur : {r['error']} | | | |")
            continue
        lines.append(
            f"| YOLO11n ONNX {variant} | {r['map50_95']:.3f} | {r['map50']:.3f} "
            f"| {r['size_mb']:.1f} | {r['latency_mean_ms']:.1f} "
            f"(p95 {r['latency_p95_ms']:.1f}) |"
        )
    return "\n".join(lines) + "\n"


def main():
    params = load_params()
    imgsz = params["export"]["imgsz"]
    runs = params["export"]["latency_runs"]
    names = params["data"]["classes"]

    results = {}
    for path in sorted(ONNX_DIR.glob("touchfree_*.onnx")):
        variant = path.stem.replace("touchfree_", "")
        try:
            map50_95, map50 = accuracy(path, imgsz)
            mean_ms, p95_ms = latency(path, names, imgsz, runs)
            results[variant] = {
                "path": str(path),
                "size_mb": path.stat().st_size / 1e6,
                "map50_95": map50_95,
                "map50": map50,
                "latency_mean_ms": mean_ms,
                "latency_p95_ms": p95_ms,
            }
        except Exception as exc:  # une variante qui echoue ne bloque pas les autres
            results[variant] = {"path": str(path), "error": str(exc)[:200]}
        print(variant, results[variant])

    yolo = read_json(REPORTS_DIR / "yolo_metrics.json")
    frcnn = read_json(REPORTS_DIR / "frcnn_metrics.json")
    REPORTS_DIR.mkdir(exist_ok=True)
    bench = {"onnx": results, "yolo_pytorch": yolo, "frcnn": frcnn}
    (REPORTS_DIR / "benchmark.json").write_text(
        json.dumps(bench, indent=2), encoding="utf-8"
    )
    table = markdown_table(results, yolo, frcnn)
    (REPORTS_DIR / "comparison.md").write_text(table, encoding="utf-8")
    print(table)

    mlflow = setup_mlflow(params["mlflow"]["experiment"])
    with mlflow.start_run(run_name="benchmark-onnx"):
        mlflow.set_tag("role", "optimisation")
        for variant, r in results.items():
            if "error" in r:
                continue
            for key in ("map50_95", "map50", "size_mb", "latency_mean_ms"):
                mlflow.log_metric(f"{variant}.{key}", r[key])
        mlflow.log_artifact(str(REPORTS_DIR / "benchmark.json"))
        mlflow.log_artifact(str(REPORTS_DIR / "comparison.md"))


if __name__ == "__main__":
    main()
