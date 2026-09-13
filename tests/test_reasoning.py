"""Tests for the reasoning layer. No GPU and no checkpoint required, because
the layer only ever consumes plain dictionaries.

These double as the worked examples in the memo: the insufficient-information
case at the bottom is the one to quote.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import reasoning
from app.scene import build_scene


def det(label, conf, box):
    return {"label": label, "confidence": conf, "box_xyxy": box}


def scene(dets, width=1000, height=800, floor=0.25):
    return build_scene(dets, width, height, floor)


# --- routing ---------------------------------------------------------------

def test_counting_question_calls_the_detector():
    r = reasoning.route("How many people are in this image?")
    assert r.needs_detection is True
    assert r.kind == reasoning.KIND_COUNT


def test_compliance_question_is_recognised():
    r = reasoning.route("Is anyone not wearing a helmet?")
    assert r.needs_detection is True
    assert r.kind == reasoning.KIND_COMPLIANCE


def test_unrelated_question_skips_the_detector():
    r = reasoning.route("What is the capital of France?")
    assert r.needs_detection is False
    assert r.kind == reasoning.KIND_NOT_IMAGE


def test_visual_question_outside_the_class_list_skips_the_detector():
    r = reasoning.route("What colour is the truck in this photo?")
    assert r.needs_detection is False
    assert r.kind == reasoning.KIND_OUT_OF_SCOPE


# --- helmet to person association -----------------------------------------

def test_helmet_inside_upper_person_box_marks_compliance():
    facts = scene([
        det("person", 0.94, [100, 100, 200, 400]),
        det("helmet", 0.88, [130, 105, 170, 140]),
    ])
    assert facts.compliant == 1
    assert facts.violations == 0
    assert facts.unknown == 0


def test_bare_head_marks_a_violation():
    facts = scene([
        det("person", 0.93, [100, 100, 200, 400]),
        det("head", 0.81, [130, 105, 170, 140]),
    ])
    assert facts.violations == 1
    assert facts.compliant == 0


def test_helmet_carried_by_a_visible_worker_is_not_credited():
    """A helmet inside a person box but nowhere near their head is being
    carried, not worn. It must not read as compliance, and the worker whose
    head is unresolved must be reported as unknown."""
    facts = scene([
        det("person", 0.92, [100, 100, 200, 400]),
        det("helmet", 0.77, [120, 370, 160, 398]),
    ])
    assert facts.compliant == 0
    assert facts.unknown == 1
    assert facts.helmets_not_worn == 1


def test_headgear_without_person_boxes_still_yields_compliance():
    """The trained model finds helmets and bare heads reliably (mAP50 0.96 and
    0.91) but almost never emits a person box (recall 0.03), because the
    source dataset labels a person on ~3 percent of workers. A helmet box is a
    head with a helmet on it and a head box is a bare head, so each one is a
    worker's status on its own."""
    facts = scene([
        det("helmet", 0.93, [130, 105, 170, 140]),
        det("helmet", 0.90, [330, 110, 372, 148]),
        det("head", 0.86, [530, 120, 568, 156]),
    ])
    assert facts.person_boxes == 0
    assert len(facts.workers) == 3
    assert facts.compliant == 2
    assert facts.violations == 1
    assert facts.unknown == 0
    guard = reasoning.guardrail(reasoning.KIND_COMPLIANCE, facts)
    assert guard.sufficient is True


def test_person_box_and_free_headgear_are_both_counted():
    facts = scene([
        det("person", 0.90, [100, 100, 200, 400]),
        det("helmet", 0.91, [130, 105, 170, 140]),   # on that person's head
        det("head", 0.84, [430, 120, 468, 156]),     # a worker with no person box
    ])
    assert facts.compliant == 1
    assert facts.violations == 1
    assert facts.unknown == 0
    assert len(facts.workers) == 2


def test_two_workers_are_scored_independently():
    facts = scene([
        det("person", 0.95, [100, 100, 200, 400]),
        det("helmet", 0.90, [130, 105, 170, 140]),
        det("person", 0.91, [300, 120, 400, 420]),
        det("head", 0.84, [330, 125, 370, 160]),
    ])
    assert facts.compliant == 1
    assert facts.violations == 1


# --- guardrail -------------------------------------------------------------

def test_no_detections_is_insufficient():
    facts = scene([])
    guard = reasoning.guardrail(reasoning.KIND_COUNT, facts)
    assert guard.sufficient is False


def test_clean_scene_is_sufficient():
    facts = scene([
        det("person", 0.95, [100, 100, 200, 400]),
        det("helmet", 0.90, [130, 105, 170, 140]),
    ])
    guard = reasoning.guardrail(reasoning.KIND_COMPLIANCE, facts)
    assert guard.sufficient is True


def test_worker_with_no_visible_head_forces_insufficient_information():
    """The memo's worked example.

    Three workers are detected. Two have a clearly associated helmet. The third
    is detected as a person but no helmet and no bare head falls inside the top
    of their box, because their head is occluded by the worker in front. The
    honest answer is not "two are compliant" and it is certainly not "one is in
    violation" -- it is that one worker's status cannot be determined.
    """
    facts = scene([
        det("person", 0.95, [100, 100, 200, 400]),
        det("helmet", 0.91, [130, 105, 170, 140]),
        det("person", 0.93, [220, 110, 320, 410]),
        det("helmet", 0.88, [250, 115, 290, 150]),
        det("person", 0.71, [330, 180, 430, 460]),  # head hidden behind a beam
    ])
    assert facts.unknown == 1
    guard = reasoning.guardrail(reasoning.KIND_COMPLIANCE, facts)
    assert guard.sufficient is False
    assert any("no helmet and no bare head" in r for r in guard.reasons)

    message = reasoning.insufficient_message(reasoning.KIND_COMPLIANCE, guard.reasons)
    assert "do not have enough information" in message


def test_all_weak_detections_are_insufficient():
    facts = scene([
        det("person", 0.31, [100, 100, 200, 400]),
        det("helmet", 0.28, [130, 105, 170, 140]),
    ])
    guard = reasoning.guardrail(reasoning.KIND_COUNT, facts)
    assert guard.sufficient is False
