"""Fetch the fine-tuned checkpoint from its public Kaggle Dataset.

Usage:
    python scripts/download_weights.py                 # -> artifacts/best.pt
    python scripts/download_weights.py --out /tmp/w.pt

Public datasets need no Kaggle credentials.
"""
import argparse
import os
import shutil
from pathlib import Path

DEFAULT_DATASET = os.getenv("WEIGHTS_KAGGLE_DATASET", "kanagavelak/ppe-rtdetr-weights")


def download(dataset: str, out: Path) -> Path:
    import kagglehub

    root = Path(kagglehub.dataset_download(dataset))
    src = next(root.rglob("best.pt"), None)
    if src is None:
        raise SystemExit(f"no best.pt inside {dataset} (looked under {root})")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="owner/slug of the Kaggle dataset")
    ap.add_argument("--out", default="artifacts/best.pt")
    args = ap.parse_args()
    path = download(args.dataset, Path(args.out))
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
