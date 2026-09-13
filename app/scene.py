"""Turn a flat list of boxes into the structured facts the reasoning layer needs.

The detector answers "what is in this image". It does not answer "who is wearing
what" -- that is a relation between two boxes, and deriving it is this module's
whole job. Counting helmets and counting people and subtracting is wrong the
moment one worker is out of frame or one helmet sits on a bench.

Association rule, stated plainly so it can be defended:
  a helmet or bare-head box belongs to a person box when most of the headgear
  box lies inside that person box, and its centre sits in the upper part of the
  person box. Of the candidates that pass, the tightest containment wins.

Every person who ends up with neither a helmet nor a bare head is reported as
UNKNOWN, never as compliant and never as a violation. That single choice is
what lets the API say "insufficient information" honestly.
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
    person_confidence: float
    status: str
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
    unassigned_helmets: int = 0
    unassigned_heads: int = 0
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
            "compliant_workers": self.compliant,
            "violations": self.violations,
            "undetermined_workers": self.unknown,
            "helmets_not_matched_to_a_person": self.unassigned_helmets,
            "bare_heads_not_matched_to_a_person": self.unassigned_heads,
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

    workers = []
    claimed = set()

    for i, person in enumerate(people):
        best = None  # (containment, kind, detection index, detection)
        for kind, pool, offset in (("helmet", helmets, 0), ("head", heads, 10_000)):
            for j, gear in enumerate(pool):
                key = offset + j
                if key in claimed:
                    continue
                containment = _containment(gear["box_xyxy"], person["box_xyxy"])
                if containment < CONTAINMENT_MIN:
                    continue
                if not _in_head_zone(gear["box_xyxy"], person["box_xyxy"]):
                    continue
                if best is None or containment > best[0]:
                    best = (containment, kind, key, gear)

        if best is None:
            workers.append(Worker(
                index=i,
                box_xyxy=person["box_xyxy"],
                person_confidence=person["confidence"],
                status=STATUS_UNKNOWN,
                note="no helmet or bare head could be associated with this person, "
                     "so their compliance cannot be determined",
            ))
            continue

        _, kind, key, gear = best
        claimed.add(key)
        workers.append(Worker(
            index=i,
            box_xyxy=person["box_xyxy"],
            person_confidence=person["confidence"],
            status=STATUS_COMPLIANT if kind == "helmet" else STATUS_VIOLATION,
            headgear=kind,
            headgear_confidence=gear["confidence"],
        ))

    unassigned_helmets = sum(1 for j in range(len(helmets)) if j not in claimed)
    unassigned_heads = sum(1 for j in range(len(heads)) if 10_000 + j not in claimed)

    facts = SceneFacts(
        image_width=image_width,
        image_height=image_height,
        confidence_floor=confidence_floor,
        counts=dict(counts),
        workers=workers,
        unassigned_helmets=unassigned_helmets,
        unassigned_heads=unassigned_heads,
        compliant=sum(1 for w in workers if w.status == STATUS_COMPLIANT),
        violations=sum(1 for w in workers if w.status == STATUS_VIOLATION),
        unknown=sum(1 for w in workers if w.status == STATUS_UNKNOWN),
        max_confidence=round(max(confidences), 4) if confidences else 0.0,
        mean_confidence=round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        detections=dets,
    )
    return facts
