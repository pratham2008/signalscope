"""Optional local Gradio demo for SignalScope."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from predict import DEVICE, predict_details
from src.core.explain import grounded_explanation, overlay, saliency_map


def analyse(image, model_path: str):
    if image is None:
        raise gr.Error("Upload an image first.")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        image.convert("RGB").save(temporary_path)
        result = predict_details(temporary_path, model_path or None)
        heatmap = saliency_map(result["model"], result["tensor"], DEVICE)
        text = grounded_explanation(result["probability_ai_generated"], result["confidence"])
        score = {"AI-generated": result["probability_ai_generated"], "Authentic / real": 1 - result["probability_ai_generated"]}
        return score, text, overlay(result["image"], heatmap)
    finally:
        temporary_path.unlink(missing_ok=True)


with gr.Blocks(title="SignalScope") as demo:
    gr.Markdown("# SignalScope\nA calibrated, probabilistic assessment — not proof of image provenance.")
    with gr.Row():
        image_input = gr.Image(type="pil", label="Image")
        heatmap_output = gr.Image(label="Model-sensitivity overlay")
    model_input = gr.Textbox(label="Checkpoint path (optional)", placeholder="model/dual/signalscope_dual_stream.pt")
    run = gr.Button("Analyse image", variant="primary")
    scores = gr.Label(label="Assessment")
    explanation = gr.Textbox(label="Explanation", lines=4)
    run.click(analyse, [image_input, model_input], [scores, explanation, heatmap_output])


if __name__ == "__main__":
    demo.launch()
