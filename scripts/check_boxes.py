"""Dessine les boites YOLO sur 8 images pour verifier la conversion."""

import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path("data/subset")


def draw_sample(img_path, label_path, names):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    for line in label_path.read_text().splitlines():
        cid, cx, cy, w, h = line.split()
        cx, cy, w, h = (float(v) for v in (cx, cy, w, h))
        x0, y0 = (cx - w / 2) * iw, (cy - h / 2) * ih
        x1, y1 = (cx + w / 2) * iw, (cy + h / 2) * ih
        draw.rectangle([x0, y0, x1, y1], outline="lime", width=4)
        draw.text((x0 + 6, y0 + 6), names[int(cid)], fill="yellow")
    return img.resize((320, 320))


def main():
    with open(ROOT / "data.yaml", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    images = sorted((ROOT / "images" / "train").glob("*.jpg"))
    picks = random.Random(0).sample(images, 8)
    sheet = Image.new("RGB", (1280, 640))
    for k, img_path in enumerate(picks):
        label = ROOT / "labels" / "train" / f"{img_path.stem}.txt"
        tile = draw_sample(img_path, label, names)
        sheet.paste(tile, ((k % 4) * 320, (k // 4) * 320))
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    sheet.save("docs/figures/boxes_check.jpg")


if __name__ == "__main__":
    main()
