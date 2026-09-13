"""Mine the test split for the model's worst images and attach measured
evidence for each failure, so the memo's root-cause analysis is grounded in
numbers rather than in what an annotated image looks like at a glance.

For every test image it matches predictions to ground truth greedily by IoU,
counts false negatives, false positives and class confusions, then measures
four image properties that commonly explain them:

  blur        variance of the Laplacian over the missed region
  scale       missed box area as a fraction of image area
  occlusion   max IoU between the missed box and any other ground-truth box
  exposure    mean luminance of the missed region

Usage:
    python scripts/failure_cases.py --weights runs/rtdetr_ppe/weights/best.pt \
        --data /kaggle/working/data/ppe/data.yaml --top 5 \
        --out artifacts/failures
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from prepare_data import CLASSES

COLOURS = {0: (0, 200, 0), 1: (0, 0, 230), 2: (230, 160, 0)}


def yolo_to_xyxy(line, width, height):
    cls, cx, cy, bw, bh = (float(v) for v in line.split()[:5])
    x1 = (cx - bw / 2) * width
    y1 = (cy - bh / 2) * height
    x2 = (cx + bw / 2) * width
    y2 = (cy + bh / 2) * height
    return int(cls), np.array([x1, y1, x2, y2], dtype=float)


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def crop_stats(image, box):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
    x2, y2 = min(x2, w), min(y2, h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return {"blur_laplacian_var": None, "mean_luminance": None}
    crop = image[y1:y2, x1:x2]
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return {
        "blur_laplacian_var": round(float(cv2.Laplacian(grey, cv2.CV_64F).var()), 2),
        "mean_luminance": round(float(grey.mean()), 2),
    }


def diagnose(record):
    """Turn measurements into a first-pass hypothesis. Verify it by eye before
    it goes in the memo. This narrows the search, it does not replace looking."""
    reasons = []
    if record["scale_fraction"] is not None and record["scale_fraction"] < 0.005:
        reasons.append("small object: under 0.5 percent of image area")
    if record["blur_laplacian_var"] is not None and record["blur_laplacian_var"] < 60:
        reasons.append("motion or focus blur: low Laplacian variance")
    if record["max_overlap_with_other_gt"] > 0.35:
        reasons.append("occlusion or crowding: heavy overlap with a neighbouring box")
    if record["mean_luminance"] is not None and record["mean_luminance"] < 55:
        reasons.append("underexposed region")
    if record["mean_luminance"] is not None and record["mean_luminance"] > 205:
        reasons.append("blown highlights or backlighting")
    if record["kind"] == "class_confusion":
        reasons.append("class confusion: predicted {} for a {}".format(
            record["predicted_class"], record["true_class"]))
    return reasons or ["no single measured cause stands out, inspect manually"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default="artifacts/failures")
    args = ap.parse_args()

    from ultralytics import RTDETR

    data_root = Path(args.data).parent
    img_dir = data_root / "images" / "test"
    lbl_dir = data_root / "labels" / "test"
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise SystemExit("no test images under {}".format(img_dir))

    model = RTDETR(args.weights)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        label_path = lbl_dir / (image_path.stem + ".txt")
        gt = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    gt.append(yolo_to_xyxy(line, w, h))

        result = model.predict(str(image_path), conf=args.conf, verbose=False)[0]
        preds = []
        for box in result.boxes:
            preds.append((int(box.cls.item()),
                          box.xyxy[0].cpu().numpy().astype(float),
                          float(box.conf.item())))

        matched_pred = set()
        errors = []
        for gi, (gcls, gbox) in enumerate(gt):
            best_j, best_iou = -1, 0.0
            for j, (pcls, pbox, _) in enumerate(preds):
                if j in matched_pred:
                    continue
                score = iou(gbox, pbox)
                if score > best_iou:
                    best_iou, best_j = score, j

            if best_iou < args.iou_match:
                kind, pred_cls = "missed_detection", None
            else:
                matched_pred.add(best_j)
                if preds[best_j][0] == gcls:
                    continue
                kind, pred_cls = "class_confusion", CLASSES[preds[best_j][0]]

            others = [b for k, (_, b) in enumerate(gt) if k != gi]
            rec = {
                "kind": kind,
                "true_class": CLASSES[gcls],
                "predicted_class": pred_cls,
                "box_xyxy": [round(v, 1) for v in gbox.tolist()],
                "best_iou": round(best_iou, 3),
                "scale_fraction": round(
                    float((gbox[2] - gbox[0]) * (gbox[3] - gbox[1]) / (w * h)), 5),
                "max_overlap_with_other_gt": round(
                    max((iou(gbox, o) for o in others), default=0.0), 3),
            }
            rec.update(crop_stats(image, gbox))
            rec["hypotheses"] = diagnose(rec)
            errors.append(rec)

        false_positives = []
        for j, (pcls, pbox, conf) in enumerate(preds):
            if j in matched_pred:
                continue
            false_positives.append({
                "kind": "false_positive",
                "predicted_class": CLASSES[pcls],
                "confidence": round(conf, 3),
                "box_xyxy": [round(v, 1) for v in pbox.tolist()],
            })

        total = len(errors) + len(false_positives)
        if total == 0:
            continue
        scored.append({
            "image": image_path.name,
            "ground_truth_count": len(gt),
            "prediction_count": len(preds),
            "error_count": total,
            "errors": errors,
            "false_positives": false_positives,
        })

    scored.sort(key=lambda r: (-r["error_count"], r["image"]))
    worst = scored[: args.top]

    for record in worst:
        path = img_dir / record["image"]
        image = cv2.imread(str(path))
        result = model.predict(str(path), conf=args.conf, verbose=False)[0]
        for box in result.boxes:
            cls = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            colour = COLOURS.get(cls, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(image, "{} {:.2f}".format(CLASSES[cls], box.conf.item()),
                        (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        for err in record["errors"]:
            x1, y1, x2, y2 = [int(v) for v in err["box_xyxy"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(image, "MISS " + err["true_class"],
                        (x1, min(image.shape[0] - 4, y2 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(str(out_dir / ("failure_" + record["image"])), image)

    (out_dir / "failure_report.json").write_text(
        json.dumps({"conf": args.conf,
                    "iou_match": args.iou_match,
                    "images_with_errors": len(scored),
                    "worst": worst}, indent=2),
        encoding="utf-8")

    print(json.dumps([{"image": r["image"], "errors": r["error_count"]} for r in worst],
                     indent=2))
    print("\nannotated images and the full report are in {}".format(out_dir))


if __name__ == "__main__":
    main()
