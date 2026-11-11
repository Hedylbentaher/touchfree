"""Conversion HaGRID -> format YOLO et controles anti-fuite."""

import json
from collections import Counter
from pathlib import Path

import yaml


def load_cfg():
    with open("params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["data"]


def clamp(v):
    return min(max(v, 0.0), 1.0)


def to_yolo_lines(entry, class_to_id):
    """[x, y, w, h] (coin haut-gauche) -> 'id cx cy w h' (centre)."""
    lines = []
    for (x, y, w, h), label in zip(entry["bboxes"], entry["labels"]):
        if label not in class_to_id:
            continue
        cx, cy = clamp(x + w / 2), clamp(y + h / 2)
        w, h = clamp(w), clamp(h)
        lines.append(f"{class_to_id[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def check_no_leak(users):
    names = list(users)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = users[a] & users[b]
            assert not common, f"FUITE: {len(common)} user_id communs {a}/{b}"


def main():
    cfg = load_cfg()
    out = Path(cfg["subset_dir"])
    class_to_id = {c: i for i, c in enumerate(cfg["classes"])}
    users, stats = {}, {}

    for split in cfg["per_class"]:
        ann = json.loads((out / f"annotations_{split}.json").read_text("utf-8"))
        lab_dir = out / "labels" / split
        lab_dir.mkdir(parents=True, exist_ok=True)
        counts, empty = Counter(), 0
        for stem, entry in ann.items():
            lines = to_yolo_lines(entry, class_to_id)
            empty += not lines
            counts.update(line.split()[0] for line in lines)
            (lab_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        users[split] = {e["user_id"] for e in ann.values()}
        stats[split] = {
            "images": len(ann),
            "empty_labels": empty,
            "boxes_per_class": {
                cfg["classes"][int(k)]: v for k, v in sorted(counts.items())
            },
        }

    check_no_leak(users)
    data_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(cfg["classes"])),
    }
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("OK : aucune fuite de user_id entre train, val et test")
    print(json.dumps({s: v["images"] for s, v in stats.items()}))


if __name__ == "__main__":
    main()
