"""SignalScope FastAPI backend."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict import DEVICE, predict_details
from src.core.explain import grounded_explanation, occlusion_saliency


MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "dual"
    / "clip_fft_resolution.pt"
)


app = FastAPI(
    title="SignalScope API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Basic health check."""

    return {
        "status": "ok",
        "model": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
    }


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
) -> dict:
    """Classify one uploaded image."""

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported image type. "
                "Use JPEG, PNG, WebP, or BMP."
            ),
        )

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="SignalScope model checkpoint is missing.",
        )

    suffix = Path(
        file.filename or ""
    ).suffix or ".img"

    temporary_path = None

    try:
        data = await file.read()

        if not data:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        with NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as handle:
            handle.write(data)
            temporary_path = Path(handle.name)

        # Validate that the uploaded bytes are actually an image.
        try:
            with Image.open(temporary_path) as image:
                image.verify()
        except (
            UnidentifiedImageError,
            OSError,
        ) as exc:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image.",
            ) from exc

        result = predict_details(
            temporary_path,
            MODEL_PATH,
        )

        response = {
            "label": result["label"],
            "confidence": result["confidence"],
            "probability_ai_generated": (
                result["probability_ai_generated"]
            ),
            "threshold": result["threshold"],
            "model": result["model_name"],
        }

        if result["model_name"] == "clip_fft_fusion":
            heatmap, evidence = occlusion_saliency(
                result["model"],
                result["clip_model"],
                result["image"],
                float(
                    result["metadata"].get(
                        "temperature",
                        1.0,
                    )
                ),
                DEVICE,
                grid=7,
                batch_size=16,
            )

            buffer = io.BytesIO()
            Image.fromarray(heatmap).save(
                buffer,
                format="PNG",
            )

            encoded_image = base64.b64encode(
                buffer.getvalue()
            ).decode("ascii")

            response["explanation_image"] = (
                f"data:image/png;base64,{encoded_image}"
            )
            response["explanation_method"] = (
                "7x7 occlusion attribution"
            )
            response["explanation_evidence"] = evidence
            response["explanation_text"] = (
                grounded_explanation(
                    result["probability_ai_generated"],
                    result["confidence"],
                    result["threshold"],
                )
            )

        return response

    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )
