"""Evaluate the fine-tuned model on the in-domain test split and, optionally,
on the out-of-distribution SH17 slice.

Reporting both is the point. The in-domain number tells you how well the model
learned this dataset; the OOD number is the closest honest proxy available for
performance on an unseen hidden test set.

Usage:
    python scripts/evaluate.py --weights runs/rtdetr_ppe/weights/best.pt \
        --data /kaggle/working/data/ppe/data.yaml \
        --ood  /kaggle/working/data/ood_sh17/data.yaml \
        --out  artifacts/metrics.json
"""
import argparse
import json
from pathlib import Path

from prepare_data import CLASSES


def run_split(model, data_yaml, split, imgsz, batch, tag):
    results = model.val(data=data_yaml, split=split, imgsz=imgsz, batch=batch,
                        plots=True, name=f"val_{tag}", exist_ok=True)
    box = results.box
    per_class = {}
    for i, name in enumerate(CLASSES):
        try:
            p, r, ap50, ap = box.class_result(i)
            per_class[name] = {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "mAP50": round(float(ap50), 4),
                "mAP50_95": round(float(ap), 4),
            }
        except Exception:
            per_class[name] = {"note": "no ground-truth instances of this class in the split"}
    return {
        "split": split,
        "data": str(data_yaml),
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ood", default=None, help="data.yaml for the SH17 OOD slice")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="artifacts/metrics.json")
    args = ap.parse_args()

    from ultralytics import RTDETR

    model = RTDETR(args.weights)
    report = {"weights": args.weights, "in_domain": run_split(
        model, args.data, "test", args.imgsz, args.batch, "in_domain")}

    if args.ood:
        report["out_of_distribution"] = run_split(
            model, args.ood, "test", args.imgsz, args.batch, "ood")
        gap = report["in_domain"]["mAP50"] - report["out_of_distribution"]["mAP50"]
        report["generalisation_gap_mAP50"] = round(gap, 4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
