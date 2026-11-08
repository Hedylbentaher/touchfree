"""Garde seulement bboxes, labels et user_id des annotations HaGRID."""

import json
import sys
import zipfile
from pathlib import Path

import yaml

KEEP = ("bboxes", "labels", "user_id")


def main(zip_path, out_dir):
    with open("params.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["data"]
    classes = set(cfg["classes"])
    splits = set(cfg["per_class"])
    out = Path(out_dir)
    written = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            p = Path(name)
            if p.suffix != ".json" or p.stem not in classes:
                continue
            if p.parent.name not in splits:
                continue
            with zf.open(name) as src:
                ann = json.load(src)
            light = {img: {k: e.get(k) for k in KEEP} for img, e in ann.items()}
            dest = out / p.parent.name / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(light), encoding="utf-8")
            written += 1
            print(f"{p.parent.name}/{p.name} : {len(light)} images")
    print(f"{written} fichiers ecrits (attendu : {len(classes) * len(splits)})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
