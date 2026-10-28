import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "soilsense_mobilenetv2.keras"
LABELS_PATH = ROOT / "class_names.json"
IMG_SIZE = (224, 224)

app = FastAPI(title="SoilSense Prediction API", version="1.0.0")

# Restrict browser requests to the public SoilSense website.
# Update this list only if the GitHub Pages address changes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://soilsenseai.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_model = None
_labels = None

def load_assets():
    global _model, _labels
    if _model is None:
        if not MODEL_PATH.exists() or not LABELS_PATH.exists():
            raise RuntimeError("Model or class_names.json is missing from the API folder.")
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        if isinstance(labels, dict):
            labels = [labels.get(str(i), labels.get(i)) for i in range(int(_model.output_shape[-1]))]
        if not isinstance(labels, list) or len(labels) != int(_model.output_shape[-1]):
            raise RuntimeError("class_names.json does not match the model output.")
        _labels = [str(x) for x in labels]
    return _model, _labels

@app.get("/")
def home():
    return {
        "service": "SoilSense Prediction API",
        "status": "ok",
        "usage": "POST an image file in the 'file' field to /predict",
        "docs": "/docs",
    }

@app.get("/health")
def health():
    try:
        model, labels = load_assets()
        return {"status": "ok", "model_loaded": model is not None, "classes": labels}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model failed to load: {type(exc).__name__}: {exc}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image.")
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="The uploaded image is empty.")
        from io import BytesIO
        with Image.open(BytesIO(raw)) as source:
            image = source.convert("RGB").resize(IMG_SIZE, resample=Image.Resampling.BILINEAR)
        model, labels = load_assets()
        # This model already contains MobileNetV2 preprocess_input.
        pixels = np.asarray(image, dtype=np.float32)
        scores = model.predict(np.expand_dims(pixels, axis=0), verbose=0)[0]
        ranked = sorted(
            [{"class": labels[i], "score": round(float(scores[i]), 6),
              "percentage": round(float(scores[i]) * 100, 2)}
             for i in range(len(scores))],
            key=lambda item: item["score"], reverse=True
        )
        return {
            "predicted_class": ranked[0]["class"],
            "confidence": ranked[0]["percentage"],
            "scores": ranked,
            "disclaimer": "Visual soil-class prediction only; not a laboratory test and not a direct measurement of pH or NPK."
        }
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The file could not be read as an image. Try JPG or PNG.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}: {exc}")
