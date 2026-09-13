"""Build an out-of-distribution test set from SH17 and remap it onto our 3 classes.

SH17 is never trained on. It comes from a different image source (Pexels stock
photography) than the construction-site training images, so measuring on it
gives an honest read on how the model behaves off its training distribution.

Source: https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection
Licence: CC BY-NC-SA 4.0 -- research use, cite the authors.

Usage:
    python scripts/prepare_ood.py \
        --root /kaggle/input/sh17-dataset-for-ppe-detection \
        --out  /kaggle/working/data/ood_sh17 \
        --limit 600
"""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from prepare_data import CLASSES, CLASS_TO_ID

# SH17 name -> our name. Everything else is dropped.
NAME_MAP = {
    "person": "person",
    "head": "head",
    "helmet": "helmet",
}


def load_sh17_names(root: Path, explicit: Path = None):
    """Read SH17's own class list rather than assuming an index order."""
    candidates = [explicit] if explicit else []
    candidates += sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))
    for candidate in candidates:
        text = candidate.read_text(encoding="utf-8", errors="replace")
        if "names" not in text:
            continue
        names = {}
        for line in text.splitlines():
            m = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if m:
                names[int(m.group(1))] = m.group(2).strip().strip("'\"").lower()
        if names:
            print(f"class list read from {candidate}")
            return names
    raise SystemExit(
        "could not find a data yaml with a names: block under the SH17 root. "
        "Open the dataset in the Kaggle file browser and pass the class order manually."
    )


def find_label_for(image_path: Path, label_dirs):
    for d in label_dirs:
        p = d / f"{image_path.stem}.txt"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=600,
                    help="cap the number of images so evaluation stays fast")
    ap.add_argument("--names", default=None,
                    help="yaml with a names: block, used when the dataset ships without one")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    (out / "images" / "test").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "test").mkdir(parents=True, exist_ok=True)

    sh17_names = load_sh17_names(root, Path(args.names) if args.names else None)
    # SH17 index -> our index, for the three classes we share.
    index_map = {}
    for idx, name in sh17_names.items():
        target = NAME_MAP.get(name.replace("_", "-"))
        if target:
            index_map[idx] = CLASS_TO_ID[target]
    if not index_map:
        raise SystemExit(f"no overlapping classes found in {sh17_names}")
    print("index map (sh17 -> ours):", index_map)

    label_dirs = [p for p in root.rglob("labels") if p.is_dir()]
    images = [p for p in root.rglob("*")
              if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "label" not in p.parts]
    images.sort()

    kept = 0
    counts = Counter()
    for image_path in images:
        if kept >= args.limit:
            break
        label_path = find_label_for(image_path, label_dirs)
        if label_path is None:
            continue

        lines = []
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            src = int(float(parts[0]))
            if src not in index_map:
                continue
            dst = index_map[src]
            lines.append(" ".join([str(dst)] + parts[1:5]))
            counts[CLASSES[dst]] += 1

        if not lines:
            continue  # an image with none of our classes teaches us nothing here

        shutil.copy2(image_path, out / "images" / "test" / image_path.name)
        (out / "labels" / "test" / f"{image_path.stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8")
        kept += 1

    (out / "data.yaml").write_text(
        "\n".join([
            f"path: {out.resolve().as_posix()}",
            "train: images/test",   # unused, ultralytics wants the key present
            "val: images/test",
            "test: images/test",
            "names:",
            *[f"  {i}: {n}" for i, n in enumerate(CLASSES)],
            "",
        ]),
        encoding="utf-8",
    )

    stats = {"images": kept, "instances": dict(counts), "source": "SH17 (CC BY-NC-SA 4.0)"}
    (out / "ood_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
