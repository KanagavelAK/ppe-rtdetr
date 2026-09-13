"""The reasoning layer: routing, structured reasoning, confidence guardrail.

Hand written on purpose. No agent framework is used or needed -- the whole
control flow is the three functions below, called in order by `answer_question`:

    route()      decide whether the detector has to run at all
    guardrail()  decide, from the detections, whether an honest answer exists
    compose()    put the facts into plain language

The guardrail runs BEFORE the language model sees anything, and it is
deterministic. A model that is asked to grade its own confidence will talk
itself into an answer; a rule that says "three of five workers have no
associated headgear, so compliance is undetermined" will not.

The language model is optional. With no API key configured the layer answers
from templates over the same structured facts, so the API is fully runnable
offline. Set ANTHROPIC_API_KEY to enable the fluent path.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger("ppe.reasoning")

MODEL_ID = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
LOW_CONFIDENCE_FLOOR = float(os.getenv("LOW_CONFIDENCE_FLOOR", "0.45"))

# ---------------------------------------------------------------------------
# Intent routing
# ---------------------------------------------------------------------------

# What the detector can actually see. A question outside this vocabulary is
# image grounded but unanswerable by us, which is a different thing from a
# question that is not about the image at all.
KNOWN_VOCABULARY = {
    "person", "people", "worker", "workers", "man", "men", "woman", "women",
    "human", "humans", "crew", "staff", "helmet", "helmets", "hardhat",
    "hard-hat", "hardhats", "hat", "hats", "head", "heads", "ppe",
    "compliance", "compliant", "safety",
}

IMAGE_REFERENCE = re.compile(
    r"\b(image|picture|photo|photograph|frame|shot|scene|here|visible|shown|"
    r"this\s+(?:site|site\s+photo)?)\b", re.I)

COMPLIANCE_PATTERN = re.compile(
    r"\b(wear|wearing|worn|without|not\s+wearing|no\s+helmet|bare\s*head|"
    r"compliance|compliant|violation|violating|unsafe|safe)\b", re.I)

COUNT_PATTERN = re.compile(r"\b(how\s+many|count|number\s+of|total)\b", re.I)

PRESENCE_PATTERN = re.compile(r"\b(is\s+there|are\s+there|any|anyone|does\s+it\s+show)\b", re.I)

# Questions that are plainly not about this image at all.
NON_VISUAL_PATTERN = re.compile(
    r"\b(who\s+(?:are|is)\s+you|what\s+model|your\s+(?:name|architecture|training)|"
    r"capital\s+of|weather\s+(?:today|tomorrow)|what\s+time|tell\s+me\s+a\s+joke|"
    r"how\s+do\s+i\s+(?:train|install|deploy)|what\s+is\s+the\s+meaning|"
    r"translate|write\s+(?:me\s+)?(?:a|some)\s+(?:poem|code|essay))\b", re.I)

# Visual attributes we can see the question is about but genuinely cannot measure.
OUT_OF_SCOPE_VISUAL = re.compile(
    r"\b(colou?r|brand|logo|text|sign\s+say|read\s+the|weather|time\s+of\s+day|"
    r"temperature|age|gender|name\s+of\s+(?:the\s+)?(?:person|worker)|emotion|"
    r"vehicle|truck|crane|ladder|scaffold|glove|vest|boot|goggle|mask)\b", re.I)

KIND_COUNT = "count"
KIND_COMPLIANCE = "compliance"
KIND_PRESENCE = "presence"
KIND_SUMMARY = "summary"
KIND_NOT_IMAGE = "not_about_the_image"
KIND_OUT_OF_SCOPE = "out_of_detector_scope"


@dataclass
class Route:
    needs_detection: bool
    kind: str
    rationale: str
    decided_by: str = "rules"


def route(question: str) -> Route:
    """Decide whether the detector has to run.

    Deterministic by design. Routing is a cheap, high-traffic decision with a
    small, closed vocabulary, so a rule set is both faster and easier to defend
    than a model call, and it cannot hallucinate a route.
    """
    q = (question or "").strip()
    if not q:
        return Route(False, KIND_NOT_IMAGE, "the question was empty")

    if NON_VISUAL_PATTERN.search(q):
        return Route(False, KIND_NOT_IMAGE,
                     "the question asks about something other than the contents "
                     "of the image, so running the detector would not inform it")

    if OUT_OF_SCOPE_VISUAL.search(q) and not COMPLIANCE_PATTERN.search(q):
        return Route(False, KIND_OUT_OF_SCOPE,
                     "the question is about the image but asks for an attribute "
                     "this detector does not predict, so no amount of detection "
                     "would answer it")

    tokens = set(re.findall(r"[a-z-]+", q.lower()))
    mentions_known = bool(tokens & KNOWN_VOCABULARY)
    mentions_image = bool(IMAGE_REFERENCE.search(q))

    if not mentions_known and not mentions_image:
        return Route(False, KIND_NOT_IMAGE,
                     "the question names nothing this detector can see and does "
                     "not refer to the image")

    if COMPLIANCE_PATTERN.search(q):
        return Route(True, KIND_COMPLIANCE,
                     "the question is about who is or is not wearing a helmet, "
                     "which needs per-person detections")
    if COUNT_PATTERN.search(q):
        return Route(True, KIND_COUNT,
                     "the question asks for a count of objects in the image")
    if PRESENCE_PATTERN.search(q):
        return Route(True, KIND_PRESENCE,
                     "the question asks whether something is present in the image")
    return Route(True, KIND_SUMMARY,
                 "the question is about the contents of the image in general")


# ---------------------------------------------------------------------------
# Confidence guardrail
# ---------------------------------------------------------------------------

@dataclass
class Guard:
    sufficient: bool
    reasons: List[str] = field(default_factory=list)


def guardrail(kind: str, facts) -> Guard:
    """Decide whether the detections support an honest answer.

    Runs before the language model and never consults it.
    """
    reasons = []

    if not facts.detections:
        reasons.append("the detector returned no objects above the confidence "
                       "floor of {:.2f}".format(facts.confidence_floor))
        return Guard(False, reasons)

    if facts.max_confidence < LOW_CONFIDENCE_FLOOR:
        reasons.append("every detection scored below {:.2f}, which is too weak to "
                       "assert anything about this image".format(LOW_CONFIDENCE_FLOOR))

    if kind == KIND_COMPLIANCE:
        if not facts.workers:
            reasons.append("no people, helmets or bare heads were detected, so "
                           "helmet compliance cannot be assessed")
        if facts.unknown > 0:
            reasons.append(
                "{} of {} detected people have no helmet and no bare head "
                "associated with them, most likely because their head is occluded, "
                "cropped or too small to resolve".format(
                    facts.unknown, len(facts.workers)))
        headgear_conf = [w.headgear_confidence for w in facts.workers
                         if w.headgear_confidence is not None]
        if headgear_conf and max(headgear_conf) < LOW_CONFIDENCE_FLOOR:
            reasons.append("every helmet and bare-head detection scored below {:.2f}, "
                           "which is too weak to state who is wearing what".format(
                               LOW_CONFIDENCE_FLOOR))

    if kind == KIND_COUNT and facts.mean_confidence < LOW_CONFIDENCE_FLOOR:
        reasons.append("the mean detection confidence of {:.2f} is too low for the "
                       "count to be reliable".format(facts.mean_confidence))

    return Guard(not reasons, reasons)


# ---------------------------------------------------------------------------
# Answer composition
# ---------------------------------------------------------------------------

PEOPLE_WORDS = re.compile(
    r"\b(people|person|persons|workers?|humans?|men|women|man|woman|crew|staff)\b", re.I)


def _deterministic_answer(question: str, kind: str, facts) -> str:
    counts = facts.counts
    if kind == KIND_COUNT:
        if PEOPLE_WORDS.search(question) and facts.workers:
            n = len(facts.workers)
            tail = "{} wearing a helmet, {} bare-headed".format(facts.compliant, facts.violations)
            if facts.unknown:
                tail += ", {} undetermined".format(facts.unknown)
            return "I count {} {} in this image ({}).".format(
                n, "person" if n == 1 else "people", tail)
        parts = ["{} {}".format(v, k if v == 1 else k + "s") for k, v in sorted(counts.items())]
        return "I detect " + (", ".join(parts) if parts else "nothing") + " in this image."
    if kind == KIND_COMPLIANCE:
        return ("Of {} people detected, {} are wearing a helmet and {} are not."
                .format(len(facts.workers), facts.compliant, facts.violations))
    if kind == KIND_PRESENCE:
        present = [k for k, v in counts.items() if v > 0]
        return ("The image contains " + ", ".join(sorted(present)) + "."
                if present else "I detect none of the objects I am trained on.")
    parts = ["{} {}".format(v, k) for k, v in sorted(counts.items())]
    return ("This looks like a work-site image containing " + ", ".join(parts) + ". "
            "{} of the {} detected people are wearing a helmet."
            .format(facts.compliant, len(facts.workers)))


SYSTEM_PROMPT = """You answer questions about a construction-site photograph.

You cannot see the photograph. You are given the structured output of an
RT-DETR object detector that was fine-tuned on three classes: helmet, head
(a human head with no helmet on it), and person. A separate deterministic step
has already matched headgear to people and has already decided that these
detections are sufficient to answer.

Rules:
- Answer only from the structured facts given. Never infer objects, attributes,
  or context that is not in them.
- Be direct and plain. Two or three sentences at most.
- Give the numbers that matter and say what they are counts of.
- Never describe your own confidence as a percentage. State what was detected.
- If the facts contain workers with status "unknown", say how many and that
  their status could not be determined."""


def _llm_answer(question: str, kind: str, facts) -> Optional[str]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import json

        import anthropic
        from pydantic import BaseModel

        class Answer(BaseModel):
            answer: str

        client = anthropic.Anthropic()
        payload = json.dumps(facts.to_dict(), indent=2, sort_keys=True)
        response = client.messages.parse(
            model=MODEL_ID,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            messages=[{
                "role": "user",
                "content": ("Question: {}\nQuestion type: {}\n\nDetector facts:\n{}"
                            .format(question, kind, payload)),
            }],
            output_format=Answer,
        )
        if response.stop_reason == "refusal":
            log.warning("model declined to answer, falling back to templates")
            return None
        return response.parsed_output.answer.strip()
    except Exception as exc:
        log.warning("language model step failed (%s), falling back to templates", exc)
        return None


def compose(question: str, kind: str, facts) -> tuple:
    """Return (answer_text, source) where source is 'llm' or 'template'."""
    text = _llm_answer(question, kind, facts)
    if text:
        return text, "llm"
    return _deterministic_answer(question, kind, facts), "template"


def insufficient_message(kind: str, reasons: List[str]) -> str:
    lead = "I do not have enough information to answer that confidently."
    if kind == KIND_NOT_IMAGE:
        lead = ("That question is not about the contents of the image, so I did "
                "not run the detector.")
    elif kind == KIND_OUT_OF_SCOPE:
        lead = ("I cannot answer that. This detector only recognises people, "
                "helmets and bare heads, so the attribute you asked about is "
                "outside what it can measure.")
    if not reasons:
        return lead
    return lead + " " + " ".join(r[0].upper() + r[1:] + "." for r in reasons)
