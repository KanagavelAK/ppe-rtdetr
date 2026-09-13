# Site Safety Compliance Detector — technical memo

Kanagavel A K · RAP pre-hackathon screening, CV + Applied ML · code: github.com/KanagavelAK/ppe-rtdetr

## 1. Domain and dataset

**Problem.** Detect construction-site workers and whether each is wearing a hard
hat, so compliance can be monitored continuously from cameras that are already
on site instead of by a supervisor walking it. Head injury is among the leading
causes of serious harm on sites; the question a safety officer asks is not "how
many helmets" but "is anyone working without one".

**Classes.** `helmet` (a head with a helmet on it), `head` (a bare head),
`person`. Helmet and head are not COCO classes. `person` was kept as the anchor
for the compliance relation and as a read on backbone transfer; section 3
explains why that decision had to be reversed.

**Source.** Safety Helmet Detection, `andrewmvd/hard-hat-detection` on Kaggle:
5,000 images, Pascal VOC XML. Licence: **`<copy the licence line from the dataset
page verbatim>`**. No relabelling. `scripts/prepare_data.py` converts to YOLO,
drops degenerate boxes and folds the label aliases (`hat`, `hard-hat`) into
`helmet`. **Evaluation-only second dataset:** SH17,
`mugheesahmad/sh17-dataset-for-ppe-detection`, CC BY-NC-SA 4.0, stock
photography, 17 classes remapped to the same three by `scripts/prepare_ood.py`.
Never trained on.

## 2. Split strategy

Image-level, hash-based: `md5(file stem)` bucketed into test / val / train at
15 / 15 / 70 percent. Two leakage reasons. The dataset has near-duplicate
frames, so a random per-object split would put near-identical images on both
sides and inflate test mAP. And because the assignment is a pure function of
the filename, re-running preparation or adding images later cannot silently
move an image from train into test. The two `samples/` images in the repo were
chosen with the same rule, so they are guaranteed held-out.

| Split | Images | helmet | head | person |
|---|---|---|---|---|
| train | 3,447 | 12,908 | 4,089 | 497 |
| val | 798 | 3,086 | 811 | 186 |
| test | 755 | 2,972 | 885 | 68 |

Class balance is severely skewed: **helmet : head : person = 26 : 8 : 1**.
`head` is the class that matters most (a missed head is a missed violation) and
it is well represented. `person` is not: the source dataset draws a person box
on roughly 3 percent of workers, which turned out to be the single most
consequential fact about this data.

## 3. Metrics, and what they do not tell you

| | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| In-domain test (755 img) | 0.627 | 0.412 | 0.628 | 0.617 |
| SH17, out of distribution (600 img) | 0.026 | 0.011 | 0.053 | 0.036 |

| Class (in-domain test) | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| helmet | 0.927 | 0.930 | **0.959** | 0.629 |
| head | 0.877 | 0.890 | **0.914** | 0.602 |
| person | 0.078 | 0.029 | **0.008** | 0.005 |

**What the headline number hides.** 0.627 is the mean of two strong classes and
one that failed. On the two non-COCO classes the model is at 0.96 / 0.91 mAP@50;
the mean over those two is 0.937. `person` collapsed, and not because of model
capacity: with a person box on only ~3 percent of workers, the loss punishes a
correct person prediction 97 percent of the time, and the model learned to stop
making them. The confusion matrix shows person ground truth going almost
entirely to background, not to another class. This is a labelling-convention
problem in the source data, and the reason `person` recall is 0.03.

**What the OOD number measures, and what it does not.** SH17 is stock
photography: large, centred, well-lit subjects, versus small heads in wide site
shots at 416 px in training. The 0.60-point gap is real domain shift. But it is
also inflated by a *label-definition* mismatch: SH17 annotates `head` on every
head including helmeted ones, whereas here `head` means bare head only, so a
correct bare-head-only model scores zero on SH17's head class by construction.
`person` scores *higher* OOD (0.077) than in-domain because SH17 labels every
person. So the OOD figure is a lower bound on generalisation, not a measurement
of it; a fair OOD test would need a dataset with the same class definitions.

**What neither measures.** mAP is a detection metric; the product question is a
compliance decision, which depends on the relation between boxes. The operating
metrics that matter are recall on `head` (0.89, so roughly one bare head in nine
is missed) and the undetermined-worker rate from section 5. I did not measure
compliance-level accuracy end to end because the dataset has no per-worker
compliance labels to score against; deriving them from the boxes would just be
scoring the association rule against itself.

## 4. Five failure cases

> Measured by `scripts/failure_cases.py` on the test split (conf 0.25, IoU 0.5).
> Annotated images are in `artifacts/failures/`. `<Fill each case from
> artifacts/failures/failure_report.json and confirm by eye.>`

**1. `hard_hat_workers360.png` — 19 errors, `<symptom>`**
Measured: box covers `<>` of image area, Laplacian variance `<>`, overlap `<>`,
mean luminance `<>`. Root cause: `<>`. What it would take to fix: `<>`.

**2. `hard_hat_workers3822.png` — 15 errors —** `<>`
**3. `hard_hat_workers429.png` — 15 errors —** `<>`
**4. `hard_hat_workers210.png` — 14 errors —** `<>`
**5. `hard_hat_workers2095.png` — 13 errors —** `<>`

Patterns across the five: `<>`. Plus one failure mode that needs no image:
**every worker whose ground truth is a `person` box is a miss** (68 of 68 in
test, recall 0.03), which is section 3's labelling problem seen per image.

## 5. The reasoning layer

![Part B: reasoning layer](../docs/architecture_part_b.svg)

**Routing.** `route()` is a deterministic rule pass over a closed vocabulary of
three classes; it is faster than a model call and cannot hallucinate a route.
Two categories skip the detector: not about the image (*capital of France*),
and about the image but outside the class list (*what colour is the truck*).
The second matters more: a system that only checked "is this about the image"
would call the detector and then invent an answer from boxes that cannot
support one.

**Structured reasoning.** `build_scene()` turns boxes into workers. Because a
`helmet` box is a head with a helmet on it and a `head` box is a bare head,
each headgear box is one worker's status. Person boxes, when present, refine
that: a helmet ≥55 percent inside a person box but with its centre below the
top 45 percent of that box is being carried, not worn, and is not credited; a
person with no headgear in their head zone is `unknown`, never compliant and
never a violation.

**Guardrail.** `guardrail()` runs before any language model and is fully
deterministic: it refuses on no detections, on every detection below 0.45, and
for compliance questions on any `unknown` worker. The model is never asked to
rate its own confidence, because a model asked to grade itself talks itself
into an answer.

**Worked insufficient-information example** (`tests/test_reasoning.py`,
`test_worker_with_no_visible_head_forces_insufficient_information`). Three
people are detected; two have a helmet inside their head zone; the third is a
0.71-confidence person with nothing resolvable on their head, occluded by the
worker in front. The API answers: *"I do not have enough information to answer
that confidently. 1 of 3 detected people have no helmet and no bare head
associated with them, most likely because their head is occluded, cropped or
too small to resolve."* Both available answers would have been wrong: "two of
three are compliant" hides a worker; "one is in violation" invents a violation
from an absence of evidence.

**No frameworks.** The control flow is four function calls in `app/main.py:ask`.
The optional language-model step is one direct Messages API call; with no key
the same facts are phrased from templates and `answer_source` says so.

## 6. Reproducibility

`artifacts/training_receipt.json`, written by `scripts/train.py`:

- Hardware: Kaggle free tier, 2 × NVIDIA Tesla T4 (15,360 MiB each), driver 580.159.04, DDP across both
- Software: Python 3.12.13, torch 2.10.0+cu128, Ultralytics 8.3.40
- Model `rtdetr-l.pt` (COCO-pretrained), 40 epochs, batch 16 (8 per GPU), imgsz 640, AdamW, lr0 1e-4, AMP on, RAM cache, seed 0, patience 12
- Wall-clock training time: **6,168 s (1.71 h)**
- Weights: `best.pt`, 66.1 MB, optimizer stripped: `python scripts/download_weights.py` or the Kaggle dataset link in the README
- Exact steps: `notebooks/kaggle_ppe_rtdetr.ipynb`, Save & Run All, no datasets attached, GPU T4 x2, Internet on

## 7. What I tried first and changed

1. **P100 → T4 x2.** Kaggle's current torch (2.10, CUDA 12.8) no longer ships
   sm_60 kernels; the P100 failed at `model.to(device)`. The notebook now probes
   compute capability and falls back to torch 2.6 if needed.
2. **Ray Tune callback crash.** Ultralytics 8.3.40 auto-registers a Ray callback
   when `ray` is importable, and Kaggle's newer `ray` removed the private
   function it calls. Training died after epoch 1 until `raytune` was disabled.
3. **Single GPU → DDP + RAM cache.** 1.71 h for 40 epochs instead of an
   estimated 3+ h.
4. **Compliance anchored on person boxes → per-headgear.** The original scene
   logic required a person box for every worker. After seeing person recall of
   0.03, that design refused nearly every compliance question on real images
   ("no people detected" with four helmets in view). The rule was inverted:
   headgear defines the worker, person boxes only refine. The insufficient-
   information behaviour was kept where it is honest (a visible person with no
   resolvable head). This was the most useful thing the evaluation taught me.
