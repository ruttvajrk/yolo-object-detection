"""Wildlife counting API (ONNX Runtime, no PyTorch needed).

    uvicorn api:app --port 8000          # docs at http://localhost:8000/docs

POST /detect   upload one image -> animal counts per species + boxes
GET  /health   model file, classes, default confidence
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from detector_onnx import CLASSES, OnnxDetector, count

WEIGHTS = Path(os.getenv("WEIGHTS", "models/best.onnx"))
# Default confidence = the threshold that gave the most accurate counts on the validation set
# (scripts/evaluate.py -> reports/evaluation.json); falls back to 0.25 if the report is missing.
_report = Path(os.getenv("EVAL_REPORT", "reports/evaluation.json"))
DEFAULT_CONF = (json.loads(_report.read_text())["counting"]["threshold_chosen_on_val"]
                if _report.exists() else 0.25)
MAX_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Wildlife counting API", version="1.0.0")
detector = OnnxDetector(WEIGHTS) if WEIGHTS.exists() else None


@app.get("/health")
def health() -> dict:
    if detector is None:
        raise HTTPException(503, f"Model file not found: {WEIGHTS}")
    return {"status": "ok", "weights": WEIGHTS.name, "classes": CLASSES,
            "default_conf": DEFAULT_CONF}


@app.post("/detect")
async def detect(file: UploadFile = File(...),
                 conf: float = Query(None, ge=0.01, le=0.99,
                                     description="confidence threshold; default = counting-tuned")) -> dict:
    if detector is None:
        raise HTTPException(503, f"Model file not found: {WEIGHTS}")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "File is not a readable image")
    t0 = time.perf_counter()
    dets = detector(img, conf=conf or DEFAULT_CONF)
    ms = (time.perf_counter() - t0) * 1000
    return {"counts": count(dets), "total": len(dets), "conf": conf or DEFAULT_CONF,
            "inference_ms": round(ms, 1), "image_size": list(img.size),
            "detections": [{"class": d.cls, "conf": d.conf, "box_xyxy": d.box} for d in dets]}
