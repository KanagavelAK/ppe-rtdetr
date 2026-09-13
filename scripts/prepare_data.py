"""Convert the Kaggle Safety Helmet Detection dataset (Pascal VOC XML) into a
YOLO-format dataset with a deterministic, image-level train/val/test split.

Source: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection
Layout expected:  <root>/images/*.png  and  <root>/annotations/*.xml

Usage:
    python scripts/prepare_data.py \
        --root /kaggle/input/hard-hat-detection \
        --out  /kaggle/working/data/ppe
"""
import argparse
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# Fixed class order. Index is baked into the weights, so never reorder this.
CLASSES = ["helmet", "head", "person"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

# The raw dataset spells the bare-head class a few different ways across files.
ALIASES = {
    "helmet": "helmet",
    "hat": "helmet",
    "hard-hat": "helmet",
    "head": "head",
    "person": "person",
}


def parse_voc(xml_path: Path):
    """Return (width, height, [(class_name, xmin, ymin, xmax, ymax), ...])."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.find("width").text))
    height = int(float(size.find("height").text))

    boxes = []
    for obj in root.findall("object"):
        raw_name = obj.find("name").text.strip().lower()
        name = ALIASES.get(raw_name)
        if name is None:
            continue  # unknown label, skip rather than silently mislabel
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        if xmax <= xmin or ymax <= ymin:
            continue  # degenerate box
        boxes.append((name, xmin, ymin, xmax, ymax))
    return width, height, boxes


def to_yolo_line(name, xmin, ymin, xmax, ymax, width, height):
    """Pascal VOC corners -> YOLO normalised cx cy w h, clipped to the image."""
    xmin = max(0.0, min(xmin, width))
    xmax = max(0.0, min(xmax, width))
    ymin = max(0.0, min(ymin, height))
    ymax = max(0.0, min(ymax, height))
    cx = (xmin + xmax) / 2.0 / width
    cy = (ymin + ymax) / 2.0 / height
    bw = (xmax - xmin) / width
    bh = (ymax - ymin) / height
    return f"{CLASS_TO_ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def split_for(stem: str, val_frac: float, test_frac: float) -> str:
    """Hash-based split.

    The split is a pure function of the file stem, so re-running this script or
    adding images later never moves an existing image between splits. That is
    what stops train/test leakage from creeping in across reruns.
    """
    digest = hashlib.md5(stem.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < test_frac:
        return "test"
    if bucket < test_frac + val_frac:
        return "val"
    return "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root holding images/ and annotations/")
    ap.add_argument("--out", required=True, help="output directory for the YOLO dataset")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    img_dir = root / "images"
    ann_dir = root / "annotations"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise SystemExit(f"expected {img_dir} and {ann_dir} to exist")

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    per_split = Counter()
    per_split_class = {s: Counter() for s in ("train", "val", "test")}
    empty_images = 0
    skipped = 0

    xmls = sorted(ann_dir.glob("*.xml"))
    if not xmls:
        raise SystemExit(f"no XML annotations found under {ann_dir}")

    for xml_path in xmls:
        stem = xml_path.stem
        image_path = next((p for p in (img_dir / f"{stem}{ext}"
                                       for ext in (".png", ".jpg", ".jpeg"))
                           if p.exists()), None)
        if image_path is None:
            skipped += 1
            continue

        width, height, boxes = parse_voc(xml_path)
        if width <= 0 or height <= 0:
            skipped += 1
            continue

        split = split_for(stem, args.val_frac, args.test_frac)
        lines = [to_yolo_line(*b, width, height) for b in boxes]
        if not lines:
            empty_images += 1  # kept on purpose: negatives teach the model restraint

        dst_img = out / "images" / split / image_path.name
        if not dst_img.exists():
            if args.copy:
                shutil.copy2(image_path, dst_img)
            else:
                try:
                    dst_img.symlink_to(image_path.resolve())
                except OSError:
                    shutil.copy2(image_path, dst_img)  # Windows without dev mode

        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        per_split[split] += 1
        for name, *_ in boxes:
            per_split_class[split][name] += 1

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {out.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            *[f"  {i}: {n}" for i, n in enumerate(CLASSES)],
            "",
        ]),
        encoding="utf-8",
    )

    stats = {
        "images_per_split": dict(per_split),
        "instances_per_split": {s: dict(c) for s, c in per_split_class.items()},
        "background_only_images": empty_images,
        "skipped_annotations": skipped,
        "split_method": "md5(file stem) -> deterministic bucket, image level",
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
    }
    (out / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {yaml_path}")


if __name__ == "__main__":
    random.seed(0)
    main()
