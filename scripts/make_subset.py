"""Construit le sous-ensemble HaGRID (512 px) sans tout telecharger."""

import io
import json
import random
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image
from remotezip import RemoteZip
from tqdm import tqdm

HOST = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/"
V1 = HOST + "hagrid/hagrid_dataset_new_554800/hagrid_dataset/{}.zip"
V2 = HOST + "hagrid_v2/hagrid_v2_zip/{}.zip"
V2_ONLY = {"one", "rock", "no_gesture"}


def zip_url(cls):
    return (V2 if cls in V2_ONLY else V1).format(cls)


def resize_min_side(img, min_side):
    w, h = img.size
    scale = min_side / min(w, h)
    if scale >= 1:
        return img
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def pick_ids(ann, available, n, max_per_user, rng):
    ids = sorted(i for i in ann if i in available)
    rng.shuffle(ids)
    per_user, kept = Counter(), []
    for img_id in ids:
        user = ann[img_id]["user_id"]
        if per_user[user] >= max_per_user:
            continue
        per_user[user] += 1
        kept.append(img_id)
        if len(kept) == n:
            break
    return kept


def main():
    with open("params.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]
    rng = random.Random(cfg["seed"])
    ann_dir = Path(cfg["annotations_dir"])
    out = Path(cfg["subset_dir"])
    kept_ann = {split: {} for split in cfg["per_class"]}

    for cls in cfg["classes"]:
        with RemoteZip(zip_url(cls)) as rz:
            members = {
                Path(n).stem: n for n in rz.namelist() if n.lower().endswith(".jpg")
            }
            for split, n in cfg["per_class"].items():
                path = ann_dir / split / f"{cls}.json"
                ann = json.loads(path.read_text(encoding="utf-8"))
                ids = pick_ids(ann, members, n, cfg["max_per_user"], rng)
                img_dir = out / "images" / split
                img_dir.mkdir(parents=True, exist_ok=True)
                for img_id in tqdm(ids, desc=f"{cls}/{split}"):
                    stem = f"{cls}_{img_id}"
                    target = img_dir / f"{stem}.jpg"
                    if not target.exists():
                        raw = rz.read(members[img_id])
                        img = Image.open(io.BytesIO(raw)).convert("RGB")
                        img = resize_min_side(img, cfg["min_side"])
                        img.save(target, quality=92)
                    e = ann[img_id]
                    kept_ann[split][stem] = {
                        "bboxes": e["bboxes"],
                        "labels": e["labels"],
                        "user_id": e["user_id"],
                    }

    for split, data in kept_ann.items():
        dest = out / f"annotations_{split}.json"
        dest.write_text(json.dumps(data), encoding="utf-8")
    print({s: len(d) for s, d in kept_ann.items()})


if __name__ == "__main__":
    main()
