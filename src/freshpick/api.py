"""Run with: uvicorn freshpick.api:app --host 127.0.0.1 --port 8000."""
import logging
import os
from io import BytesIO
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from .predict import Predictor
from .storage import StorageRequest, storage_guidance
from .shelf_life import shelf_life_reference

app = FastAPI(title="FreshPick inference", version="0.3.0")
_predictor = None
_lock = Lock()
MAX_BYTES = 20 * 1024 * 1024


@app.post("/predict")
def predict(file: UploadFile | None = File(None), image: UploadFile | None = File(None)):
    file = image or file
    if file is None:
        raise HTTPException(422, "Choose an image to scan")
    global _predictor
    payload = file.file.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise HTTPException(413, "Image exceeds 20 MiB")
    try:
        image = Image.open(BytesIO(payload))
        if image.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
            raise HTTPException(415, "Use JPEG, PNG, WebP, or BMP")
        image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(400, "Invalid or unreadable image") from error
    try:
        with _lock:
            if _predictor is None:
                _predictor = Predictor(
                    os.environ.get("FRESHPICK_CHECKPOINT", "runs/mobilenetv3-finetuned/best.pt"),
                    identifier_checkpoint=os.environ.get("FRESHPICK_IDENTIFIER", "runs/identifier/best.pt") or None)
            return _predictor.predict(image)
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error
    except Exception as error:
        logging.getLogger(__name__).exception("FreshPick inference failed")
        raise HTTPException(503, "Model inference unavailable; check server logs") from error
    finally:
        image.close()


@app.post("/storage-guidance")
def guidance(request: StorageRequest):
    return storage_guidance(request)


@app.get("/shelf-life-reference")
def reference_dataset():
    return shelf_life_reference()


# Register static routing last so API routes and /docs remain available.
_ui_dir = Path(__file__).resolve().parents[2] / "ui"
if _ui_dir.is_dir():
    app.mount("/", StaticFiles(directory=_ui_dir, html=True), name="ui")
