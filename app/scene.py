"""Turn a flat list of boxes into the structured facts the reasoning layer needs.

The detector answers "what is in this image". It does not answer "who is wearing
what" -- that is a relation between two boxes, and deriving it is this module's
whole job. Counting helmets and counting people and subtracting is wrong the
moment one worker is out of frame or one helmet sits on a bench.

Two facts about the training data shape the rule:

  1. In the source dataset a `helmet` box is a head WITH a helmet on it and a
     `head` box is a head WITHOUT one. Each headgear box is therefore one
     worker's compliance status in its own right.
  2. `person` boxes are annotated on only ~3 percent of workers (497 person vs
     12,908 helmet instances in the training split), so the fine-tuned model
     rarely predicts them (test recall 0.03). Compliance cannot be anchored on
     person boxes; if it were, almost every question would be refused.

So: every headgear box becomes a worker. Person boxes, when present, refine
that: a headgear box that lies inside a person box but not in its head zone
is judged not worn (a helmet carried at the hip) and is excluded, and a person
box with no headgear inside its head zone is a worker whose status is UNKNOWN,
never compliant and never a violation. That last case is what lets the API
say "insufficient information" honestly.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

# Fraction of the headgear box that must fall inside the person box.
CONTAINMENT_MIN = 0.55
# The headgear centre must sit within this fraction of the person box height,
# measured from the top. Generous, because people bend, crouch and lean.
HEAD_ZONE_FRACTION = 0.45

STATUS_COMPLIANT = "compliant"
STATUS_VIOLATION = "violation"
STATUS_UNKNOWN = "unknown"


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _containment(inner, outer) -> float:
    """Fraction of `inner` that lies inside `outer`."""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    inner_area = _area(inner)
    return overlap / inner_area if inner_area > 0 else 0.0


def _in_head_zone(headgear, person) -> bool:
    centre_y = (headgear[1] + headgear[3]) / 2.0
    height = person[3] - person[1]
    if height <= 0:
        return False
    return (centre_y - person[1]) / height <= HEAD_ZONE_FRACTION


@dataclass
class Worker:
    index: int
    box_xyxy: List[float]
    status: str
    person_confidence: Optional[float] = None
    headgear: Optional[str] = None
    headgear_confidence: Optional[float] = None
    note: str = ""


@dataclass
class SceneFacts:
    image_width: int
    image_height: int
    confidence_floor: float
    counts: dict = field(default_factory=dict)
    workers: List[Worker] = field(default_factory=list)
    person_boxes: int = 0
    helmets_not_worn: int = 0
    heads_ignored: int = 0
    compliant: int = 0
    violations: int = 0
    unknown: int = 0
    max_confidence: float = 0.0
    mean_confidence: float = 0.0
    detections: List[dict] = field(default_factory=list)

    def to_dict(self):
        return {
            "image_size": {"width": self.image_width, "height": self.image_height},
            "confidence_floor": self.confidence_floor,
            "counts": self.counts,
            "people_detected": len(self.workers),
            "person_boxes": self.person_boxes,
            "compliant_workers": self.compliant,
            "violations": self.violations,
            "undetermined_workers": self.unknown,
            "helmets_seen_but_not_worn": self.helmets_not_worn,
            "max_confidence": self.max_confidence,
            "mean_confidence": self.mean_confidence,
            "workers": [
                {
                    "id": w.index,
                    "status": w.status,
                    "headgear": w.headgear,
                    "headgear_confidence": w.headgear_confidence,
                    "person_confidence": w.person_confidence,
                    "box_xyxy": w.box_xyxy,
                    "note": w.note,
                }
                for w in self.workers
            ],
            "detections": self.detections,
        }


def build_scene(detections, image_width: int, image_height: int,
                confidence_floor: float) -> SceneFacts:
    """Group raw detections into per-worker compliance facts."""
    dets = [d.to_dict() if hasattr(d, "to_dict") else dict(d) for d in detections]
    counts = Counter(d["label"] for d in dets)
    confidences = [d["confidence"] for d in dets]

    people = [d for d in dets if d["label"] == "person"]
    helmets = [d for d in dets if d["label"] == "helmet"]
    heads = [d for d in dets if d["label"] == "head"]
    gear_pool = [("helmet", g) for g in helmets] + [("head", g) for g in heads]

    workers = []
    claimed = set()

    # Pass 1: people with a headgear box in their head zone, or with none at all.
    for person in people:
        best = None  # (containment, key, kind, detection)
        for key, (kind, gear) in enumerate(gear_pool):
            if key in claimed:
                continue
            containment = _containment(gear["box_xyxy"], person["box_xyxy"])
            if containment < CONTAINMENT_MIN or not _in_head_zone(gear["box_xyxy"], person["box_xyxy"]):
                continue
            if best is None or containment > best[0]:
                best = (containment, key, kind, gear)

        if best is None:
            workers.append(Worker(
                index=len(workers),
                box_xyxy=person["box_xyxy"],
                status=STATUS_UNKNOWN,
                person_confidence=person["confidence"],
                note="a person was detected but no helmet or bare head could be "
                     "resolved on them, so their compliance cannot be determined",
            ))
            continue

        _, key, kind, gear = best
        claimed.add(key)
        workers.append(Worker(
            index=len(workers),
            box_xyxy=person["box_xyxy"],
            status=STATUS_COMPLIANT if kind == "helmet" else STATUS_VIOLATION,
            person_confidence=person["confidence"],
            headgear=kind,
            headgear_confidence=gear["confidence"],
        ))

    # Pass 2: headgear with no person box. A helmet lying inside a visible
    # person box but off their head is being carried, not worn: exclude it.
    # Headgear with no person box around it at all is a worker whose person box
    # the detector missed, which is the common case for this model.
    helmets_not_worn = 0
    heads_ignored = 0
    for key, (kind, gear) in enumerate(gear_pool):
        if key in claimed:
            continue
        inside_someone = any(_containment(gear["box_xyxy"], p["box_xyxy"]) >= CONTAINMENT_MIN
                             for p in people)
        if inside_someone:
            if kind == "helmet":
                helmets_not_worn += 1
            else:
                heads_ignored += 1
            continue
        workers.append(Worker(
            index=len(workers),
            box_xyxy=gear["box_xyxy"],
            status=STATUS_COMPLIANT if kind == "helmet" else STATUS_VIOLATION,
            headgear=kind,
            headgear_confidence=gear["confidence"],
            note="status read from the headgear box alone; no person box was detected",
        ))

    facts = SceneFacts(
        image_width=image_width,
        image_height=image_height,
        confidence_floor=confidence_floor,
        counts=dict(counts),
        workers=workers,
        person_boxes=len(people),
        helmets_not_worn=helmets_not_worn,
        heads_ignored=heads_ignored,
        compliant=sum(1 for w in workers if w.status == STATUS_COMPLIANT),
        violations=sum(1 for w in workers if w.status == STATUS_VIOLATION),
        unknown=sum(1 for w in workers if w.status == STATUS_UNKNOWN),
        max_confidence=round(max(confidences), 4) if confidences else 0.0,
        mean_confidence=round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        detections=dets,
    )
    return facts
