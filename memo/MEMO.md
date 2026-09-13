# Site Safety Compliance Detector — technical memo

> Two pages maximum. Fill every `<>` from your actual run. Numbers you did not
> measure do not go in here. The failure section is graded harder than the
> metrics section, so write it first and give it the most room.

## 1. Domain and dataset

**Problem.** Detect construction-site workers and whether each one is wearing a
hard hat, so that hard-hat compliance can be monitored continuously from
cameras that are already installed instead of by periodic manual inspection.

**Classes.** `helmet`, `head` (a human head with no helmet on it), `person`.
Helmet and head are not COCO classes. `person` is kept deliberately: it is the
anchor the compliance relation is computed against, and it gives a clean read on
how much the pretrained backbone transferred.

**Source.** Safety Helmet Detection, `andrewmvd/hard-hat-detection` on Kaggle,
5,000 images with Pascal VOC XML annotations. Licence: `<copy the licence line
from the dataset page verbatim>`. No relabelling was done; the conversion to
YOLO format is `scripts/prepare_data.py`, which drops degenerate boxes and
normalises the label aliases the raw XML uses for the same class.

**Second dataset, evaluation only.** SH17, `mugheesahmad/sh17-dataset-for-ppe-detection`,
CC BY-NC-SA 4.0, remapped onto the same three classes by
`scripts/prepare_ood.py`. It was never trained on. See section 3.

## 2. Split strategy

Hash-based, at image level: the split assignment is `md5(file stem)` bucketed
into test / val / train at `<15>` / `<15>` / `<70>` percent.

Two reasons, both about leakage. The dataset contains near-duplicate frames, so
a random per-object or per-augmentation split would put visually near-identical
images on both sides of the boundary and inflate test mAP. And because the
assignment is a pure function of the filename, re-running preparation or adding
images later cannot silently move an image from train into test.

| Split | Images | helmet | head | person |
|---|---|---|---|---|
| train | `<>` | `<>` | `<>` | `<>` |
| val | `<>` | `<>` | `<>` | `<>` |
| test | `<>` | `<>` | `<>` | `<>` |

Class balance is skewed: `<describe the real ratio>`. `head` is the minority
class and is also the class that matters most, because a missed `head` is a
missed violation. Section 4 returns to this.

## 3. Metrics, and what they do not tell you

| | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| In-domain test | `<>` | `<>` | `<>` | `<>` |
| SH17, out of distribution | `<>` | `<>` | `<>` | `<>` |

Per class on the in-domain test split:

| Class | Precision | Recall | mAP@50 |
|---|---|---|---|
| helmet | `<>` | `<>` | `<>` |
| head | `<>` | `<>` | `<>` |
| person | `<>` | `<>` | `<>` |

**What the in-domain number measures.** How well the model learned this dataset,
including its camera angles, its weather and its annotation conventions.

**What it does not measure.** Whether the model works on a site it has not seen.
That is why SH17 is reported alongside: different image source, different
framing, same three classes, never trained on. The gap of `<>` mAP@50 points is
the part of the in-domain score that is dataset-specific rather than general.

**What neither measures.** mAP is a detection metric, and the product question
is a compliance decision. A model can score well on mAP and still make
compliance errors, because compliance depends on the *relation* between a helmet
box and a person box. The relevant operating metrics are per-class recall on
`head` (a missed violation) and the rate of undetermined workers (section 5).
`<If you measured a compliance-level accuracy, put it here; if not, say you did
not and say why.>`

## 4. Five failure cases

> Pull these from `artifacts/failures/failure_report.json`, which measures blur,
> scale, occlusion and exposure for every miss. Confirm each hypothesis by
> looking at the annotated image before writing it up. Include the image.

**1. `<filename>` — `<one-line symptom>`**
Measured: box covers `<>` of image area, Laplacian variance `<>`, overlap with
nearest neighbour `<>`, mean luminance `<>`.
Root cause: `<>`.
What it would take to fix: `<>`.

**2. `<filename>` —**
**3. `<filename>` —**
**4. `<filename>` —**
**5. `<filename>` —**

Patterns across the five: `<e.g. four of five misses are under 0.5 percent of
image area, so the dominant failure mode is scale, not class confusion. Say what
that implies about where the model should and should not be deployed.>`

## 5. The reasoning layer

![Part B: reasoning layer](../docs/architecture_part_b.svg)

The `/ask` request runs top to bottom through the diagram. Every purple box is
a deterministic rule that runs before any model is consulted; the only model
call on the happy path is the last one. The two grey side exits are correct
outputs, not errors.

**Routing.** `app/reasoning.py:route` classifies the question into six kinds
with a deterministic rule pass, and returns whether the detector is needed.
Routing is a high-traffic decision over a closed vocabulary of three classes, so
rules are both faster than a model call and impossible to hallucinate a route
with. Two categories skip the detector entirely: questions not about the image
(`What is the capital of France?`) and questions about the image that ask for an
attribute the detector does not predict (`What colour is the truck?`). The
second category matters more than the first: it is where a system that only
checked "is this about the image" would call the detector and then invent an
answer from boxes that cannot support one.

**Structured reasoning.** `app/scene.py` converts flat boxes into per-worker
facts. A helmet or bare head is associated with a person when at least 55
percent of the headgear box falls inside the person box and its centre sits in
the top 45 percent of that box. The thresholds are generous because workers
bend and crouch. Every person is assigned `compliant`, `violation`, or
`unknown`, and unmatched helmets and heads are counted separately, because they
mean either a helmet that is not on anybody or a worker the person detector
missed.

**Guardrail.** `app/reasoning.py:guardrail` runs before any language model call
and is fully deterministic. It refuses when there are no detections, when every
detection is below `<0.45>` confidence, or, for compliance questions, when any
worker is `unknown` or any headgear box went unmatched. The model is never asked
to rate its own confidence, because a model asked to grade itself will talk
itself into an answer.

**Worked insufficient-information example** (`tests/test_reasoning.py`, last
test). Three people are detected. Two have a clearly contained helmet. The third
is detected as a person at 0.71 confidence with no helmet and no bare head in
the top of their box, because their head is occluded by the worker in front. The
API answers:

> I do not have enough information to answer that confidently. 1 of 3 detected
> people have no helmet and no bare head associated with them, most likely
> because their head is occluded, cropped or too small to resolve.

Two answers were available and both would have been wrong: "two of three are
compliant" hides a worker, and "one worker is in violation" invents a violation
from an absence of evidence.

**No frameworks.** The control flow is four function calls in
`app/main.py:ask`. There is no LangChain, LangGraph, CrewAI or AutoGen anywhere
in the repository. The optional language-model step is one direct Messages API
call, and the API runs correctly with no API key set at all.

## 6. Reproducibility

Paste `artifacts/training_receipt.json` here, or the fields from it:

- Hardware: `<Kaggle free tier, NVIDIA P100 16 GB>`
- Model: `rtdetr-l.pt`, Ultralytics `<version>`
- Epochs `<40>`, batch `<8>`, imgsz `<640>`, optimizer `<AdamW>`, lr0 `<1e-4>`, seed `<0>`
- Wall-clock training time: `<>`
- Weights: `<direct download link or repo path>`

## 7. What I tried first and changed

`<If nothing changed, say so. If something did, say what and why. A pivot is
useful signal, not a penalty.>`
