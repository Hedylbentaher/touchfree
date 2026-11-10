"""Porte qualite (quality gate) : decide si le modele merite la production.

1. Choisit la variante a servir : INT8 si elle perd peu de precision, sinon FP32.
2. Verifie les seuils de params.yaml (mAP minimale, latence maximale).
3. Si c'est accepte : copie le modele dans models/touchfree.onnx, ecrit
   models/metadata.json et l'enregistre dans le MLflow Model Registry
   avec l'alias « production ».
Si c'est refuse : le script s'arrete avec le code 1 (utile en CI).

Lancement :
    python scripts/promote.py
"""

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

MODELS_DIR = Path("models")
BENCH = Path("reports/benchmark.json")


def choose_variant(onnx_results, max_drop):
    fp32 = onnx_results.get("fp32", {})
    int8 = onnx_results.get("int8", {})
    if "map50_95" not in fp32:
        raise SystemExit("REFUSE : pas de resultat FP32 dans reports/benchmark.json")
    if "map50_95" in int8 and fp32["map50_95"] - int8["map50_95"] <= max_drop:
        return "int8", int8
    return "fp32", fp32


def register(path, name, metrics, variant):
    """Enregistre le modele dans MLflow et lui donne l'alias « production »."""
    import mlflow
    import onnx
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("touchfree")
    with mlflow.start_run(run_name=f"promotion-{variant}"):
        mlflow.set_tag("role", "promotion")
        mlflow.log_param("variant", variant)
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "path"})
        mlflow.onnx.log_model(
            onnx.load(str(path)), artifact_path="model", registered_model_name=name
        )
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    version = max(int(v.version) for v in versions)
    try:
        client.set_registered_model_alias(name, "production", str(version))
    except Exception:  # serveurs MLflow anciens : etapes au lieu des alias
        client.transition_model_version_stage(name, str(version), "Production")
    return str(version)


def main():
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    gate = params["gate"]
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    variant, result = choose_variant(bench["onnx"], gate["max_int8_drop"])

    print(f"Variante choisie : {variant}")
    print(
        f"  mAP50-95 test : {result['map50_95']:.3f} (minimum {gate['min_map50_95']})"
    )
    print(
        f"  latence CPU   : {result['latency_mean_ms']:.1f} ms "
        f"(maximum {gate['max_latency_ms']})"
    )
    if result["map50_95"] < gate["min_map50_95"]:
        print("REFUSE : precision insuffisante.")
        sys.exit(1)
    if result["latency_mean_ms"] > gate["max_latency_ms"]:
        print("REFUSE : modele trop lent.")
        sys.exit(1)

    target = MODELS_DIR / "touchfree.onnx"
    shutil.copy(result["path"], target)
    metrics = {
        "map50_95": round(result["map50_95"], 4),
        "map50": round(result["map50"], 4),
        "latency_mean_ms": round(result["latency_mean_ms"], 1),
        "size_mb": round(result["size_mb"], 1),
    }
    version = "local"
    if os.getenv("MLFLOW_TRACKING_URI"):
        try:
            version = register(target, gate["model_name"], metrics, variant)
        except Exception as exc:  # le modele reste utilisable meme sans Registry
            print(f"Attention : enregistrement MLflow impossible ({exc})")
    metadata = {
        "names": params["data"]["classes"],
        "imgsz": params["export"]["imgsz"],
        "variant": variant,
        "version": version,
        "metrics": metrics,
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"ACCEPTE : {target} (version {version}) est le modele de production.")


if __name__ == "__main__":
    main()
