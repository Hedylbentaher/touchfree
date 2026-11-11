"""Baseline Faster R-CNN (torchvision) sur les memes donnees, suivi dans MLflow.

Objectif : comparer un detecteur « deux etapes » classique a YOLO11.
Lancement :
    python -m src.models.train_frcnn
"""

import json
import random
import time
from pathlib import Path

from src.utils.mlflow_utils import flatten, load_params, setup_mlflow

ROOT = Path("data/subset")
REPORTS_DIR = Path("reports")


def read_yolo_labels(label_path, width, height):
    """Lignes YOLO 'id cx cy w h' (0-1) -> boites [x1, y1, x2, y2] en pixels."""
    boxes, labels = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            cid, cx, cy, w, h = line.split()
            cx, cy = float(cx) * width, float(cy) * height
            w, h = float(w) * width, float(h) * height
            if w < 1 or h < 1:
                continue
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            labels.append(int(cid) + 1)  # 0 est reserve au fond (background)
    return boxes, labels


class YoloTxtDataset:
    """Lit les images et les labels YOLO produits par src/data/dataset.py."""

    def __init__(self, split, train):
        self.images = sorted((ROOT / "images" / split).glob("*.jpg"))
        self.labels_dir = ROOT / "labels" / split
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        path = self.images[idx]
        img = Image.open(path).convert("RGB")
        width, height = img.size
        boxes, labels = read_yolo_labels(
            self.labels_dir / f"{path.stem}.txt", width, height
        )
        if self.train and random.random() < 0.5:  # miroir horizontal seulement
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [[width - x2, y1, width - x1, y2] for x1, y1, x2, y2 in boxes]
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return to_tensor(img), target


def collate(batch):
    return tuple(zip(*batch))


def build_model(num_classes, min_size):
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")  # pre-entraine COCO
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    model.transform.min_size = (min_size,)
    return model


def evaluate(model, loader, device):
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([img.to(device) for img in images])
            preds = [{k: v.cpu() for k, v in o.items()} for o in outputs]
            metric.update(preds, list(targets))
    result = metric.compute()
    return float(result["map"]), float(result["map_50"])


def measure_latency(model, dataset, device, runs=50):
    import torch

    model.eval()
    times = []
    with torch.no_grad():
        for i in range(min(runs, len(dataset))):
            img, _ = dataset[i]
            img = img.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model([img])
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return sum(times[5:]) / max(len(times) - 5, 1)


def main():
    import torch
    from torch.utils.data import DataLoader

    params = load_params()
    cfg = params["train"]["frcnn"]
    names = params["data"]["classes"]
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = YoloTxtDataset("train", train=True)
    val_set = YoloTxtDataset("val", train=False)
    test_set = YoloTxtDataset("test", train=False)
    loader_args = {"batch_size": cfg["batch"], "num_workers": 2, "collate_fn": collate}
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    model = build_model(len(names), cfg["min_size"]).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        trainable,
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])

    mlflow = setup_mlflow(params["mlflow"]["experiment"])
    with mlflow.start_run(run_name="faster-rcnn-mobilenetv3"):
        mlflow.set_tags({"model_family": "faster_rcnn", "role": "baseline"})
        mlflow.log_params(flatten("", cfg))
        start = time.time()
        best_map, best_state = -1.0, None
        for epoch in range(cfg["epochs"]):
            model.train()
            total, steps = 0.0, 0
            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss = sum(model(images, targets).values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total, steps = total + loss.item(), steps + 1
            scheduler.step()
            val_map, val_map50 = evaluate(model, val_loader, device)
            mlflow.log_metrics(
                {"train_loss": total / steps, "val_map50_95": val_map},
                step=epoch,
            )
            print(f"epoque {epoch + 1}: loss={total / steps:.3f} val_mAP={val_map:.3f}")
            if val_map > best_map:
                best_map = val_map
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        test_map, test_map50 = evaluate(model, test_loader, device)
        latency = measure_latency(model, test_set, device)
        report = {
            "map50_95": test_map,
            "map50": test_map50,
            "latency_ms": latency,
            "device": device.type,
            "train_minutes": (time.time() - start) / 60,
            "epochs_done": cfg["epochs"],
        }
        mlflow.log_metrics(
            {
                "test_map50_95": test_map,
                "test_map50": test_map50,
                "latency_ms": latency,
                "train_minutes": report["train_minutes"],
            }
        )
        REPORTS_DIR.mkdir(exist_ok=True)
        out = REPORTS_DIR / "frcnn_metrics.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out))

    print(f"Faster R-CNN termine : mAP50-95 test = {test_map:.3f}")


if __name__ == "__main__":
    main()
