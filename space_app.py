"""Hugging Face Spaces entry point (Gradio SDK, free CPU tier).

One process serves both:
    /            a small Gradio demo page (upload, detect, ask)
    /detect      the FastAPI endpoints from app/main.py, unchanged
    /ask
    /docs        Swagger UI

Locally:  python space_app.py   ->  http://localhost:7860
"""
import io
import os

os.environ.setdefault("WEIGHTS_KAGGLE_DATASET", "kanagavelak/ppe-rtdetr-weights")

import gradio as gr
import uvicorn
from PIL import Image, ImageDraw

from app import reasoning
from app.detector import DEFAULT_CONF, detector
from app.main import app as api
from app.scene import build_scene

COLOURS = {"helmet": "#2ecc71", "head": "#e74c3c", "person": "#3498db"}
SAMPLES = [p for p in ("samples/site_01.png", "samples/site_02.png") if os.path.exists(p)]


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def _draw(image: Image.Image, detections) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for d in detections:
        x1, y1, x2, y2 = d.box_xyxy
        colour = COLOURS.get(d.label, "#ffffff")
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), f"{d.label} {d.confidence:.2f}", fill=colour)
    return out


def run_detect(image, confidence):
    if image is None:
        return None, {"error": "upload an image first"}
    detections, w, h, ms = detector.predict(_png_bytes(image), conf=confidence)
    counts = {}
    for d in detections:
        counts[d.label] = counts.get(d.label, 0) + 1
    return _draw(image, detections), {
        "image": {"width": w, "height": h},
        "count": len(detections),
        "counts_by_class": counts,
        "inference_ms": ms,
        "detections": [d.to_dict() for d in detections],
    }


def run_ask(image, question, confidence):
    """Same four steps as POST /ask, in the same order."""
    decision = reasoning.route(question or "")
    routing = {"needs_detection": decision.needs_detection, "question_kind": decision.kind,
               "rationale": decision.rationale}
    if not decision.needs_detection:
        return (reasoning.insufficient_message(decision.kind, []),
                {"routing": routing, "detector_called": False, "answer_source": "template"})
    if image is None:
        return "This question needs an image, but none was uploaded.", {"routing": routing}

    detections, w, h, ms = detector.predict(_png_bytes(image), conf=confidence)
    facts = build_scene(detections, w, h, confidence)
    guard = reasoning.guardrail(decision.kind, facts)
    if guard.sufficient:
        answer, source = reasoning.compose(question, decision.kind, facts)
    else:
        answer, source = reasoning.insufficient_message(decision.kind, guard.reasons), "template"
    return answer, {
        "sufficient_information": guard.sufficient,
        "routing": routing,
        "detector_called": True,
        "guardrail_reasons": guard.reasons,
        "answer_source": source,
        "evidence": {k: v for k, v in facts.to_dict().items() if k != "detections"},
        "inference_ms": ms,
    }


with gr.Blocks(title="Site Safety Compliance") as demo:
    gr.Markdown(
        "## Site Safety Compliance detector\n"
        "RT-DETR fine-tuned for **helmet** / **head** (bare) / **person**, with a hand-written "
        "reasoning layer that refuses when the detections do not support an answer. "
        "API: [`/docs`](docs) · `POST /detect` · `POST /ask`"
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Site image")
            confidence = gr.Slider(0.05, 0.9, value=DEFAULT_CONF, step=0.05, label="Confidence threshold")
            if SAMPLES:
                gr.Examples(SAMPLES, inputs=image, label="Held-out test images")
            detect_btn = gr.Button("Detect", variant="primary")
            question = gr.Textbox(label="Question about the image",
                                  value="Is anyone not wearing a helmet?")
            gr.Examples([["Is anyone not wearing a helmet?"], ["How many people are in this image?"],
                         ["What colour is the truck?"], ["What is the capital of France?"]],
                        inputs=question, label="Try these")
            ask_btn = gr.Button("Ask")
        with gr.Column():
            annotated = gr.Image(label="Detections")
            answer = gr.Textbox(label="Answer", lines=3)
            details = gr.JSON(label="Response (same fields as the API)")

    detect_btn.click(run_detect, [image, confidence], [annotated, details])
    ask_btn.click(run_ask, [image, question, confidence], [answer, details])

app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
