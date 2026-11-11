"""Fine-tuning de YOLO11 sur le sous-ensemble HaGRID, suivi dans MLflow.

Lancement (depuis la racine du depot) :
    python -m src.models.train_yolo
"""

import csv
import json
import shutil
import time
from pathlib import Path

from src.utils.mlflow_utils import clean_key, flatten, load_params, setup_mlflow

DATA_YAML = Path("data/subset/data.yaml")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("docs/figures")


def log_epoch_curves(mlflow, results_csv):
    """Envoie chaque colonne de results.csv a MLflow, epoque par epoque."""
    with open(results_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row = {k.strip(): v for k, v in row.items()}
        step = int(float(row.pop("epoch")))
        for key, value in row.items():
            try:
                mlflow.log_metric(clean_key(key), float(value), step=step)
            except ValueError:
                continue
    return len(rows)


def evaluate_on_test(weights, imgsz, names):
    """Evalue le meilleur modele sur le split TEST (jamais vu pendant le train)."""
    from ultralytics import YOLO

    metrics = YOLO(weights).val(
        data=str(DATA_YAML),
        split="test",
        imgsz=imgsz,
        batch=16,
        plots=True,
        project="runs",
        name="yolo11_test",
        exist_ok=True,
    )
    per_class = {names[i]: float(m) for i, m in enumerate(metrics.box.maps)}
    return {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
        "per_class_map50_95": per_class,
        "plots_dir": str(metrics.save_dir),
    }


def main():
    from ultralytics import YOLO, settings

    params = load_params()
    cfg = params["train"]["yolo"]
    names = params["data"]["classes"]
    settings.update({"mlflow": False})  # on gere MLflow nous-memes

    mlflow = setup_mlflow(params["mlflow"]["experiment"])
    with mlflow.start_run(run_name=f"yolo11-{Path(cfg['model']).stem}"):
        mlflow.set_tags({"model_family": "yolo11", "role": "modele_avance"})
        mlflow.log_params(flatten("", cfg))
        mlflow.log_artifact("params.yaml")
        mlflow.log_artifact("data/subset/stats.json")

        start = time.time()
        model = YOLO(cfg["model"])
        model.train(
            data=str(DATA_YAML),
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            patience=cfg["patience"],
            seed=cfg["seed"],
            lr0=cfg["lr0"],
            deterministic=True,
            project="runs",
            name="yolo11_train",
            exist_ok=True,
            **cfg["augment"],
        )
        train_minutes = (time.time() - start) / 60
        save_dir = Path(model.trainer.save_dir)
        best = Path(model.trainer.best)

        epochs_done = log_epoch_curves(mlflow, save_dir / "results.csv")
        test = evaluate_on_test(best, cfg["imgsz"], names)

        mlflow.log_metrics(
            {
                "test_map50_95": test["map50_95"],
                "test_map50": test["map50"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
                "train_minutes": train_minutes,
                "epochs_done": epochs_done,
            }
        )
        for cls, value in test["per_class_map50_95"].items():
            mlflow.log_metric(f"test_map50_95.{cls}", value)

        MODELS_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)
        shutil.copy(best, MODELS_DIR / "yolo11_best.pt")
        report = {"train_minutes": train_minutes, "epochs_done": epochs_done, **test}
        out = REPORTS_DIR / "yolo_metrics.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(out))
        mlflow.log_artifact(str(best), artifact_path="weights")
        test_dir = Path(test["plots_dir"])
        for png in list(save_dir.glob("*.png")) + list(test_dir.glob("*.png")):
            mlflow.log_artifact(str(png), artifact_path="plots")

        # Figures reprises dans le rapport LaTeX.
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        for fig in (
            save_dir / "results.png",
            test_dir / "confusion_matrix_normalized.png",
        ):
            if fig.exists():
                shutil.copy(fig, FIGURES_DIR / fig.name)

    print(f"YOLO11 termine : mAP50-95 test = {test['map50_95']:.3f}")


if __name__ == "__main__":
    main()
