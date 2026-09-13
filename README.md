# Site Safety Compliance API

RT-DETR fine-tuned on construction-site imagery to detect **person**, **helmet**
and **head** (a human head with no helmet on it), served through FastAPI, with a
hand-written reasoning layer that answers natural-language questions about an
image and refuses when the detections do not support an answer.

Two of the three classes are outside COCO, so an off-the-shelf pretrained
checkpoint cannot do this task.

## Why this problem

Head injuries are among the leading causes of serious harm on construction
sites, and hard-hat compliance is checked today by a supervisor walking the
site. Cameras are already installed on most sites. This turns an occasional
manual spot check into a continuous one, and answers the question a safety
officer actually asks, which is not "how many helmets are visible" but
"is anyone working without one".

That distinction is the engineering content of this project. Counting helmets
and counting people and subtracting gives the wrong answer the moment a helmet
is sitting on a bench or a worker is partly out of frame. The reasoning layer
associates each helmet with a person by spatial containment, and reports any
worker it cannot resolve as undetermined rather than guessing.

## Layout

```
scripts/prepare_data.py    Pascal VOC XML -> YOLO, deterministic image-level split
scripts/prepare_ood.py     SH17 -> a remapped out-of-distribution test set
scripts/train.py           RT-DETR fine-tune, writes a reproducibility receipt
scripts/evaluate.py        mAP / precision / recall on in-domain and OOD splits
scripts/failure_cases.py   mines the worst images and measures why they failed
app/detector.py            RT-DETR inference wrapper
app/scene.py               helmet-to-person association, per-worker compliance
app/reasoning.py           intent routing, confidence guardrail, answer composition
app/main.py                FastAPI endpoints
tests/test_reasoning.py    routing, association and guardrail tests, no GPU needed
notebooks/kaggle_ppe_rtdetr.ipynb   self-contained Kaggle notebook: datasets, training, eval, export
notebooks/build_kaggle_notebook.py  regenerates the notebook from the scripts above
```

## Data

| | Source | Role |
|---|---|---|
| Training | [Safety Helmet Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection), 5,000 images, Pascal VOC XML, classes helmet / head / person | train, val, in-domain test |
| Generalisation check | [SH17](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection), 8,099 images, CC BY-NC-SA 4.0 | out-of-distribution test only, never trained on |

The split is a hash of the image file stem, so re-running the preparation
script never moves an image between splits and re-running it after adding data
cannot leak a training image into the test set.

SH17 is scraped from stock photography and the training set is real site
imagery, so the gap between the two mAP numbers is a direct measurement of how
much of the model's accuracy is dataset-specific.

## Training on the Kaggle free tier

Upload `notebooks/kaggle_ppe_rtdetr.ipynb` as a new Kaggle notebook. Set
Accelerator GPU T4 x2 and Internet on; no datasets need attaching, both are
fetched by `kagglehub` inside the notebook. The P100 also works, but Kaggle's
current torch build has dropped sm_60 kernels, so the notebook's first cell
detects that and installs torch 2.6 (cu126) before training. RT-DETR is attention heavy:
batch 8 per GPU at 640 px is the largest that fits reliably on a 16 GB card, so
the notebook uses batch 16 across the two T4s with DDP. Launch through
**Save Version, Save and Run All** so a disconnect does not kill the run.

The notebook embeds every script in `scripts/` and `app/`; after changing any
of them run `python notebooks/build_kaggle_notebook.py` to regenerate it. The
equivalent shell steps are:

```bash
python scripts/prepare_data.py --root /kaggle/input/hard-hat-detection --out /kaggle/working/data/ppe
python scripts/train.py --data /kaggle/working/data/ppe/data.yaml --epochs 40 --batch 16 \
    --device 0,1 --cache ram --project /kaggle/working/runs --name rtdetr_ppe
```

Run two epochs first and multiply the reported seconds per epoch out before
committing the full budget. `training_receipt.json` records the GPU, driver,
every hyperparameter and the wall-clock time. Copy it into the memo verbatim.

## Evaluation

```bash
python scripts/evaluate.py --weights runs/rtdetr_ppe/weights/best.pt \
    --data data/ppe/data.yaml --ood data/ood_sh17/data.yaml --out artifacts/metrics.json

python scripts/failure_cases.py --weights runs/rtdetr_ppe/weights/best.pt \
    --data data/ppe/data.yaml --top 8 --out artifacts/failures
```

The failure miner ranks test images by error count, then for every miss it
measures Laplacian variance over the missed region (blur), box area as a
fraction of the image (scale), overlap with neighbouring ground-truth boxes
(occlusion) and mean luminance (exposure). It proposes a cause from those
measurements. Confirm each one by eye before it goes in the memo.

## Weights

The fine-tuned checkpoint is not committed (198 MB). Download it and place it
at `artifacts/best.pt`, or set `MODEL_WEIGHTS` to wherever you put it.

- Direct download: **TODO: paste the Kaggle Dataset / GitHub Release link here**
- Or regenerate it: run `notebooks/kaggle_ppe_rtdetr.ipynb` on Kaggle
  (Save & Run All, GPU T4 x2) and take `best.pt` from the Output tab.

`samples/` holds two images from the held-out test split for trying the
endpoints below.

## Running the API

```bash
pip install -r requirements.txt
cp .env.example .env          # point MODEL_WEIGHTS at the checkpoint
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive docs are at `http://localhost:8000/docs`.

With Docker:

```bash
docker build -t ppe-api . && docker run -p 8000:8000 --env-file .env ppe-api
```

The reasoning layer runs without an `ANTHROPIC_API_KEY`. Without one it answers
from templates over the same structured facts, so every endpoint is fully
exercisable offline. With one set, the final phrasing step is a single direct
Messages API call. No agent framework is used anywhere.

This is deliberate. Routing and the guardrail are the decisions that matter,
and both are deterministic and run before any model is called. The language
model only rephrases facts that have already been judged sufficient, so the
API's correctness does not depend on it, and a reviewer without a key sees
identical routing, evidence and refusals with `answer_source: "template"`.

## Endpoints

### `POST /detect`

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@samples/site_01.png" -F "confidence=0.25"
```

```json
{
  "request_id": "3f9a21c0b7de",
  "image": {"width": 1024, "height": 683},
  "confidence_threshold": 0.25,
  "count": 4,
  "counts_by_class": {"person": 2, "helmet": 1, "head": 1},
  "detections": [
    {"label": "person", "confidence": 0.9412, "box_xyxy": [102.4, 98.7, 214.9, 402.1]},
    {"label": "helmet", "confidence": 0.9078, "box_xyxy": [131.2, 104.5, 172.8, 141.0]},
    {"label": "person", "confidence": 0.8933, "box_xyxy": [301.0, 118.2, 408.6, 421.7]},
    {"label": "head",   "confidence": 0.8241, "box_xyxy": [332.5, 124.0, 371.9, 160.3]}
  ],
  "inference_ms": 84.3
}
```

### `POST /ask`

Compliance question, answerable:

```bash
curl -X POST http://localhost:8000/ask \
  -F "file=@samples/site_01.png" \
  -F "question=Is anyone not wearing a helmet?"
```

```json
{
  "request_id": "7c1de4a90b22",
  "question": "Is anyone not wearing a helmet?",
  "answer": "Yes. Of the 2 people detected, 1 is wearing a helmet and 1 is not.",
  "sufficient_information": true,
  "routing": {
    "needs_detection": true,
    "question_kind": "compliance",
    "rationale": "the question is about who is or is not wearing a helmet, which needs per-person detections",
    "decided_by": "rules"
  },
  "detector_called": true,
  "guardrail_reasons": [],
  "answer_source": "llm",
  "evidence": {
    "counts": {"person": 2, "helmet": 1, "head": 1},
    "compliant_workers": 1,
    "violations": 1,
    "undetermined_workers": 0,
    "workers": [
      {"id": 0, "status": "compliant", "headgear": "helmet", "headgear_confidence": 0.9078},
      {"id": 1, "status": "violation", "headgear": "head", "headgear_confidence": 0.8241}
    ]
  },
  "inference_ms": 81.6
}
```

Question that does not need the detector:

```bash
curl -X POST http://localhost:8000/ask -F "question=What is the capital of France?"
```

```json
{
  "question": "What is the capital of France?",
  "answer": "That question is not about the contents of the image, so I did not run the detector.",
  "sufficient_information": false,
  "routing": {"needs_detection": false, "question_kind": "not_about_the_image"},
  "detector_called": false,
  "answer_source": "template"
}
```

Question the detections cannot honestly support:

```json
{
  "question": "Is anyone not wearing a helmet?",
  "answer": "I do not have enough information to answer that confidently. 1 of 3 detected people have no helmet and no bare head associated with them, most likely because their head is occluded, cropped or too small to resolve.",
  "sufficient_information": false,
  "detector_called": true,
  "guardrail_reasons": ["1 of 3 detected people have no helmet and no bare head associated with them, ..."]
}
```

## How the reasoning layer decides

1. **Route.** A deterministic rule pass classifies the question into one of six
   kinds. Questions naming nothing the detector can see, and questions about
   visual attributes outside the three classes, both skip detection entirely.
   Routing is a cheap, closed-vocabulary decision, so a rule set is faster than
   a model call and cannot hallucinate a route.
2. **Detect and structure.** The detector runs, then `app/scene.py` matches each
   helmet or bare head to a person by containment plus a head-zone check and
   assigns every person one of three statuses, the third of which is
   *undetermined*.
3. **Guard.** A deterministic check, before any language model call, decides
   whether the facts support an answer. Empty detections, an all-weak scene, or
   any worker whose headgear could not be resolved makes a compliance question
   unanswerable.
4. **Compose.** Only if the guard passes does the language model turn the facts
   into a sentence, and it is instructed to use nothing but those facts.

The guardrail is deliberately not the model's own confidence estimate. A model
asked to grade itself will talk itself into an answer. A rule that says three of
five workers have no resolvable headgear will not.

## Tests

```bash
python -m pytest tests -q
```

Twelve tests cover routing, the association rule including a helmet lying on
the ground that must not be credited to a worker, and the guardrail's
insufficient-information path. None of them need a GPU or a checkpoint.

## Limits

- Three classes only. Vests, gloves, harnesses and boots are not detected, and
  the API says so rather than guessing.
- Compliance is inferred from a spatial relation between two boxes, not from
  recognising a helmet on a head. A helmet held at chest height by a standing
  worker can read as compliant.
- Training images are predominantly daytime outdoor construction scenes. Indoor
  industrial and low-light footage is outside the training distribution, which
  is what the SH17 number partially measures.
