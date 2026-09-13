"""Generate a self-contained Kaggle notebook for this repo.

The repo has no public remote, so the notebook cannot git-clone it. Instead
every script it needs is embedded as a %%writefile cell, read from the real
source files at generation time so the notebook can never drift from the code.

Usage:
    python notebooks/build_kaggle_notebook.py
    -> notebooks/kaggle_ppe_rtdetr.ipynb   (upload this to Kaggle)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_ppe_rtdetr.ipynb"

EMBEDDED_FILES = [
    "scripts/__init__.py",
    "scripts/prepare_data.py",
    "scripts/prepare_ood.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/failure_cases.py",
    "app/__init__.py",
    "app/detector.py",
    "app/scene.py",
    "app/reasoning.py",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n")}


def writefile_cell(rel_path):
    body = (ROOT / rel_path).read_text(encoding="utf-8")
    target = f"/kaggle/working/repo/{rel_path}"
    if not body.strip():
        # %%writefile rejects an empty cell body, so create empty files directly
        return code(f'from pathlib import Path\nPath("{target}").parent.mkdir(parents=True, exist_ok=True)\n'
                    f'Path("{target}").touch()\nprint("created empty", "{target}")')
    return code(f"%%writefile {target}\n{body}")


cells = []

cells.append(md("""
# Site Safety Compliance Detector — RT-DETR fine-tune on Kaggle

Fine-tunes RT-DETR to detect **person**, **helmet** and **head** (a bare head with
no helmet) on construction-site imagery, evaluates it in-domain and on an
out-of-distribution dataset, mines failure cases, and packages the checkpoint
for the FastAPI service in the repo.

**Notebook settings (right-hand panel):**

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** (recommended). P100 also works: Kaggle's current torch build has dropped sm_60 support, so cell 1 detects that and installs a compatible torch first. |
| Internet | **ON** — needed for pip, `kagglehub`, and the `rtdetr-l.pt` base weights |
| Datasets to attach | **None.** Both datasets are downloaded by code below via `kagglehub`. |

Run the real training through **Save Version → Save & Run All** so a browser
disconnect does not kill the run. Everything is written under `/kaggle/working`.
"""))

cells.append(md("""
## 1. Environment

Installs the pinned dependencies, then checks that the preinstalled torch was
compiled for this GPU. Kaggle's torch 2.10 + CUDA 12.8 image no longer includes
sm_60 kernels, so on a P100 the model fails with `CUDA error: no kernel image
is available`. If that mismatch is detected, a torch build that still supports
the card is installed **before** torch is imported into this kernel.
"""))
cells.append(code("""
import os, sys, subprocess, json
from pathlib import Path
os.makedirs("/kaggle/working/repo", exist_ok=True)
os.chdir("/kaggle/working/repo")

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

pip("ultralytics==8.3.40", "kagglehub")   # no numpy/opencv pins: Kaggle ships numpy 2 and cv2 already
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

# Probe in a subprocess so torch is not yet imported here if it has to be replaced.
probe = subprocess.run([sys.executable, "-c",
    "import torch; cap = torch.cuda.get_device_capability(); "
    "print('sm_%d%d' % cap); print(' '.join(torch.cuda.get_arch_list())); print(torch.__version__)"],
    capture_output=True, text=True)
gpu_sm, arch_list, torch_ver = (probe.stdout.strip().split("\\n") + ["", "", ""])[:3]
print(f"gpu {gpu_sm} | torch {torch_ver} compiled for: {arch_list}")

if gpu_sm and gpu_sm not in arch_list.split():
    print(f"{gpu_sm} is not supported by the preinstalled torch; installing torch 2.6.0 (cu126), ~2.5 GB ...")
    pip("--force-reinstall", "--no-deps",
        "torch==2.6.0", "torchvision==0.21.0",
        "--index-url", "https://download.pytorch.org/whl/cu126")
    pip("ultralytics==8.3.40")   # re-satisfy deps the --no-deps install skipped

import torch, ultralytics
sm = "sm_%d%d" % torch.cuda.get_device_capability()
assert sm in torch.cuda.get_arch_list(), f"{sm} still unsupported by torch {torch.__version__}: {torch.cuda.get_arch_list()}"
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpu", torch.cuda.get_device_name(0),
      "| ultralytics", ultralytics.__version__)
"""))

cells.append(md("""
## 2. Run configuration

Edit here, nowhere else. Set `DRY_RUN = True` the first time to measure
seconds per epoch, then multiply it out before committing to `EPOCHS`.
RT-DETR is attention heavy: **batch 8 per GPU at 640 px is the largest that
fits on a 16 GB card**. With two GPUs the batch is doubled and Ultralytics
trains with DDP across both; the per-card load stays at 8.
"""))
cells.append(code("""
DRY_RUN   = False   # True -> 2 epochs into runs/dryrun, then stop
EPOCHS    = 40
N_GPU     = torch.cuda.device_count()
for i in range(N_GPU):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}  {torch.cuda.get_device_properties(i).total_memory/2**30:.1f} GB")
if N_GPU < 2:
    print(f"WARNING: only {N_GPU} GPU visible. Training will still run, ~2x slower. Pick 'GPU T4 x2' next time.")
DEVICE    = ",".join(str(i) for i in range(N_GPU)) or "cpu"   # "0,1" -> Ultralytics relaunches under DDP across both cards
BATCH     = 8 * max(N_GPU, 1)                                 # 8 per GPU is the 16 GB ceiling for RT-DETR-L at 640
WORKERS   = os.cpu_count()                           # Kaggle gives 4 cores; the dataloader is the bottleneck otherwise
CACHE     = "ram"                                    # ~4 GB of decoded images; Kaggle has 30 GB RAM
IMGSZ     = 640
LR0       = 1e-4
SEED      = 0
OOD_LIMIT = 600     # SH17 images to evaluate on; keeps the OOD pass under a few minutes

WORK  = "/kaggle/working"
DATA  = f"{WORK}/data/ppe"
OOD   = f"{WORK}/data/ood_sh17"
RUNS  = f"{WORK}/runs"
ARTS  = f"{WORK}/artifacts"
RUN_NAME = "dryrun" if DRY_RUN else "rtdetr_ppe"
RUN_EPOCHS = 2 if DRY_RUN else EPOCHS
BEST = f"{RUNS}/{RUN_NAME}/weights/best.pt"
os.makedirs(ARTS, exist_ok=True)
print(json.dumps({k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}, indent=2, default=str))
"""))

cells.append(md("""
## 3. Repository code

The repo is embedded below so the notebook is self-contained. These cells are
generated from the real source files by `notebooks/build_kaggle_notebook.py`;
do not edit them here.
"""))
for rel in EMBEDDED_FILES:
    cells.append(writefile_cell(rel))

cells.append(md("""
## 4. Datasets

Downloaded by code, no manual attachment. Both are public Kaggle datasets:

- `andrewmvd/hard-hat-detection` — 5,000 images, Pascal VOC XML. **Training, validation, in-domain test.**
- `mugheesahmad/sh17-dataset-for-ppe-detection` — 8,099 stock-photo images, CC BY-NC-SA 4.0. **Out-of-distribution test only, never trained on.**
"""))
cells.append(code("""
import kagglehub
from pathlib import Path

HARDHAT_ROOT = Path(kagglehub.dataset_download("andrewmvd/hard-hat-detection"))
SH17_ROOT    = Path(kagglehub.dataset_download("mugheesahmad/sh17-dataset-for-ppe-detection"))

print("hard-hat root:", HARDHAT_ROOT)
print("  images:", len(list((HARDHAT_ROOT / "images").glob("*"))) if (HARDHAT_ROOT / "images").is_dir() else "images/ not found")
print("  annotations:", len(list((HARDHAT_ROOT / "annotations").glob("*.xml"))) if (HARDHAT_ROOT / "annotations").is_dir() else "annotations/ not found")
print("sh17 root:", SH17_ROOT)
for p in sorted(SH17_ROOT.iterdir()):
    print("  ", p.name, "(dir)" if p.is_dir() else f"{p.stat().st_size/1e3:.0f} KB")
"""))

cells.append(md("## 5. Prepare the training data\n\nPascal VOC → YOLO, with a deterministic `md5(file stem)` image-level split so reruns never leak train images into test."))
cells.append(code("""
!python scripts/prepare_data.py --root "{HARDHAT_ROOT}" --out "{DATA}" --val-frac 0.15 --test-frac 0.15
"""))

cells.append(md("""
## 6. Train

`rtdetr-l.pt` is fetched automatically by Ultralytics on first use. AMP is on
(required to fit batch 8), image caching is off (Kaggle RAM is tighter than its
disk). A `training_receipt.json` with hardware, hyperparameters and wall-clock
time is written next to the weights.
"""))
cells.append(code("""
import shutil
if Path(RUNS, RUN_NAME).exists():
    shutil.rmtree(Path(RUNS, RUN_NAME))   # a leftover run must never be mistaken for this one
    print("removed stale run directory", Path(RUNS, RUN_NAME))

!python scripts/train.py \\
    --data "{DATA}/data.yaml" \\
    --model rtdetr-l.pt \\
    --epochs {RUN_EPOCHS} --batch {BATCH} --imgsz {IMGSZ} --lr0 {LR0} --optimizer AdamW \\
    --patience 12 --workers {WORKERS} --seed {SEED} --device {DEVICE} --cache {CACHE} \\
    --project "{RUNS}" --name {RUN_NAME}
"""))
cells.append(code("""
assert Path(BEST).exists(), f"training did not produce {BEST}"
receipt_path = Path(RUNS) / RUN_NAME / "training_receipt.json"
if not receipt_path.exists():
    # train.py did not reach its receipt step (e.g. the DDP parent crashed after
    # the workers had already saved best.pt). Rebuild it from Ultralytics' own files.
    import csv, yaml
    run_dir = Path(RUNS) / RUN_NAME
    rows = list(csv.DictReader(open(run_dir / "results.csv"))) if (run_dir / "results.csv").exists() else []
    train_args = yaml.safe_load(open(run_dir / "args.yaml")) if (run_dir / "args.yaml").exists() else {}
    last = {k.strip(): v.strip() for k, v in rows[-1].items()} if rows else {}
    receipt = {
        "status": "reconstructed",
        "note": "train.py exited before writing the receipt; values below come from results.csv and args.yaml",
        "hardware": {"gpu": [torch.cuda.get_device_name(i) for i in range(N_GPU)], "torch": torch.__version__},
        "model": train_args.get("model"), "epochs": train_args.get("epochs"), "epochs_completed": len(rows),
        "batch": train_args.get("batch"), "imgsz": train_args.get("imgsz"), "lr0": train_args.get("lr0"),
        "optimizer": train_args.get("optimizer"), "seed": train_args.get("seed"), "device": str(train_args.get("device")),
        "wall_clock_seconds": float(last.get("time", 0)) or None,
        "final_val_mAP50": last.get("metrics/mAP50(B)"), "final_val_mAP50_95": last.get("metrics/mAP50-95(B)"),
        "weights": BEST,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print("WARNING: receipt was reconstructed; check the end of the training cell output for the error.\\n")
print(receipt_path.read_text())
if DRY_RUN:
    print("\\nDRY_RUN is on. Read seconds/epoch above, set DRY_RUN = False and EPOCHS, then Save & Run All.")
"""))

cells.append(md("""
## 7. Out-of-distribution test set (SH17)

SH17 has 17 classes; only `person`, `head` and `helmet` are kept and remapped
onto this model's class ids. The class order is read from the dataset's own
yaml when it ships one; otherwise the published SH17 order is used as a
fallback and the index map is printed so it can be checked by eye.
"""))
cells.append(code("""
SH17_PUBLISHED_ORDER = ["person", "head", "face", "glasses", "face-mask-medical", "face-guard",
                        "ear", "earmuffs", "hands", "gloves", "foot", "shoes", "safety-vest",
                        "tools", "helmet", "medical-suit", "safety-suit"]

has_yaml = any(True for p in list(SH17_ROOT.rglob("*.yaml")) + list(SH17_ROOT.rglob("*.yml"))
               if "names" in p.read_text(encoding="utf-8", errors="replace"))
names_arg = ""
if not has_yaml:
    fallback = Path(WORK) / "sh17_names.yaml"
    fallback.write_text("names:\\n" + "".join(f"  {i}: {n}\\n" for i, n in enumerate(SH17_PUBLISHED_ORDER)))
    names_arg = f'--names "{fallback}"'
    print("no yaml with a names: block in the dataset; using the published SH17 class order")

!python scripts/prepare_ood.py --root "{SH17_ROOT}" --out "{OOD}" --limit {OOD_LIMIT} {names_arg}
"""))

cells.append(md("## 8. Evaluate\n\nmAP / precision / recall per class on the in-domain test split and on SH17. The gap between the two is the dataset-specific part of the in-domain score."))
cells.append(code("""
!python scripts/evaluate.py \\
    --weights "{BEST}" \\
    --data "{DATA}/data.yaml" \\
    --ood  "{OOD}/data.yaml" \\
    --imgsz {IMGSZ} --batch {BATCH} \\
    --out  "{ARTS}/metrics.json"
"""))

cells.append(md("## 9. Failure mining\n\nWorst test images by error count, each miss measured for blur, scale, occlusion and exposure. Confirm every hypothesis by looking at the annotated image before it goes in the memo."))
cells.append(code("""
!python scripts/failure_cases.py \\
    --weights "{BEST}" \\
    --data "{DATA}/data.yaml" \\
    --top 8 \\
    --out "{ARTS}/failures"
"""))
cells.append(code("""
import matplotlib.pyplot as plt
import cv2

report = json.load(open(f"{ARTS}/failures/failure_report.json"))
worst = report["worst"][:4]
fig, axes = plt.subplots(max(len(worst), 1), 1, figsize=(10, 6 * max(len(worst), 1)))
for ax, rec in zip(axes if len(worst) > 1 else [axes], worst):
    img = cv2.cvtColor(cv2.imread(f"{ARTS}/failures/failure_{rec['image']}"), cv2.COLOR_BGR2RGB)
    ax.imshow(img); ax.axis("off")
    hyp = "; ".join(h for e in rec["errors"] for h in e["hypotheses"])
    ax.set_title(f"{rec['image']} — {rec['error_count']} errors\\n{hyp[:160]}", fontsize=9)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""
## 10. End-to-end check of the reasoning layer

Runs the same code path as the API's `POST /ask` — route → detect → associate →
guardrail → compose — on a test image, offline. No API key is needed; without
one the answer comes from templates over the same structured facts.
"""))
cells.append(code("""
os.environ["MODEL_WEIGHTS"] = BEST
sys.path.insert(0, "/kaggle/working/repo")
import importlib
for m in ("app.detector", "app.scene", "app.reasoning"):
    if m in sys.modules: importlib.reload(sys.modules[m])
from app.detector import Detector
from app.scene import build_scene
from app import reasoning

det = Detector(weights=BEST, imgsz=IMGSZ)
sample = sorted(Path(f"{DATA}/images/test").glob("*"))[0]
image_bytes = sample.read_bytes()

for question in ["Is anyone not wearing a helmet?",
                 "How many people are in this image?",
                 "What colour is the truck?",
                 "What is the capital of France?"]:
    decision = reasoning.route(question)
    print("\\nQ:", question)
    print("   route:", decision.kind, "| detector needed:", decision.needs_detection)
    if not decision.needs_detection:
        print("   A:", reasoning.insufficient_message(decision.kind, []))
        continue
    dets, w, h, ms = det.predict(image_bytes, conf=0.25)
    facts = build_scene(dets, w, h, 0.25)
    guard = reasoning.guardrail(decision.kind, facts)
    if not guard.sufficient:
        print("   A:", reasoning.insufficient_message(decision.kind, guard.reasons))
    else:
        answer, source = reasoning.compose(question, decision.kind, facts)
        print(f"   A ({source}):", answer)
    print("   evidence:", {k: facts.to_dict()[k] for k in ("counts", "compliant_workers", "violations", "undetermined_workers")})
"""))

cells.append(md("""
## 11. Package the artifacts

Everything the API and the memo need lands in `/kaggle/working/artifacts`,
plus a single zip. Download from the notebook's **Output** tab, then:

- `best.pt` -> `artifacts/best.pt` in the repo
- `samples/` -> `samples/` in the repo (the README curl examples use them)
- `metrics.json`, `training_receipt.json`, `split_stats.json`, `failures/` -> fill the memo

then start the API with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
"""))
cells.append(code("""
import shutil
shutil.copy(BEST, f"{ARTS}/best.pt")
shutil.copy(f"{RUNS}/{RUN_NAME}/weights/last.pt", f"{ARTS}/last.pt")
shutil.copy(f"{RUNS}/{RUN_NAME}/training_receipt.json", ARTS)
shutil.copy(f"{DATA}/split_stats.json", ARTS)
if Path(f"{OOD}/ood_stats.json").exists():
    shutil.copy(f"{OOD}/ood_stats.json", ARTS)
for plot in ("results.png", "confusion_matrix.png", "PR_curve.png"):
    src = Path(RUNS) / RUN_NAME / plot
    if src.exists():
        shutil.copy(src, ARTS)

# Two real test images so the README curl examples work on a clean checkout.
os.makedirs(f"{ARTS}/samples", exist_ok=True)
for i, img in enumerate(sorted(Path(f"{DATA}/images/test").glob("*"))[:2], 1):
    shutil.copy(img, f"{ARTS}/samples/site_{i:02d}{img.suffix}")

shutil.make_archive(f"{WORK}/ppe_artifacts", "zip", ARTS)
for p in sorted(Path(ARTS).rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size/1e6:8.1f} MB  {p.relative_to(ARTS)}")
print(f"\\nzip: {WORK}/ppe_artifacts.zip  {Path(f'{WORK}/ppe_artifacts.zip').stat().st_size/1e6:.1f} MB")
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells, {len(EMBEDDED_FILES)} embedded files)")
