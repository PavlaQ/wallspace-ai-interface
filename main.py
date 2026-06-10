"""
WallSpace AI inference service.

A tiny, dependency-light FastAPI service that runs SegFormer (ADE20K) wall
segmentation on CPU via onnxruntime. Designed for Google Cloud Run:

  - model is baked into the image at build time (no runtime dependency on HF),
  - model + a warm-up inference run once at startup,
  - one endpoint per model so it can grow into a shared "AI backend"
    (called by the Next shop AND by n8n over HTTP).

Returns a PNG mask (white = wall) sized to the uploaded image. The browser
keeps doing all the geometry/warping/compositing — only the heavy model moved
here.
"""

import io
import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

# --- Config -----------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "model.onnx")
API_SECRET = os.environ.get("API_SECRET")  # if set, every request must send it
# Match the model's training resolution: 512 for b0–b4, 640 for b5.
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", "512"))
WALL_CLASS = 0                              # ADE20K: class index 0 == "wall"

# ImageNet normalization (SegFormer preprocessing).
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- Model ------------------------------------------------------------------

_so = ort.SessionOptions()
# Match the Cloud Run vCPU count (set CPU=2 in the console).
_so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(MODEL_PATH, sess_options=_so, providers=["CPUExecutionProvider"])
_input_name = session.get_inputs()[0].name


def _preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None, ...]  # HWC -> NCHW
    return np.ascontiguousarray(arr, dtype=np.float32)


def _upsample(score: np.ndarray, w: int, h: int) -> np.ndarray:
    """Bilinearly resize a low-res score map to the full photo size."""
    return np.asarray(
        Image.fromarray(score.astype(np.float32), mode="F").resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    )


def _wall_mask(img: Image.Image) -> bytes:
    """Run the model and return a PNG (white = wall) at the original size.

    Quality matters more than the binary argmax: we take the wall score and the
    best competing-class score, upsample BOTH to the photo size, and decide per
    full-res pixel. The wall boundary (incl. sloped ceilings / skosy) follows the
    real edge instead of a blocky 128px grid scaled up after the fact.
    """
    w, h = img.size
    logits = session.run(None, {_input_name: _preprocess(img)})[0][0]  # [150, H/4, W/4]
    wall = logits[WALL_CLASS]                                          # [H/4, W/4]
    others = np.delete(logits, WALL_CLASS, axis=0).max(axis=0)         # best non-wall class
    is_wall = _upsample(wall, w, h) > _upsample(others, w, h)          # smooth full-res decision
    mask = Image.fromarray(np.where(is_wall, 255, 0).astype(np.uint8))
    buf = io.BytesIO()
    mask.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# --- App --------------------------------------------------------------------

app = FastAPI(title="WallSpace AI inference")


@app.on_event("startup")
def _warmup() -> None:
    # First inference is slow (graph alloc/optimization). Do it now so the first
    # real request after a cold start isn't extra slow on top of the boot time.
    _wall_mask(Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (0, 0, 0)))


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
) -> Response:
    if API_SECRET and x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="bad api key")
    try:
        img = Image.open(io.BytesIO(await image.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image")
    png = _wall_mask(img)
    return Response(content=png, media_type="image/png")
