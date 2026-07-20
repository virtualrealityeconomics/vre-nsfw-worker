"""NSFW body-part detection via NudeNet v3 (the MIT-licensed `nudenet` package — onnxruntime only,
NO Ultralytics code at runtime). `detect(path)` returns [{class, score, box}] per image; we
aggregate to a per-class MAX per image, then MAX across a video's frames. The gate (config.verdict)
thresholds each `*_EXPOSED` class.

LICENSING: the `nudenet` PACKAGE is MIT; only the bundled `320n.onnx` WEIGHTS derive from YOLOv8
(AGPL). This worker is kept OPEN-SOURCE and at arm's length from vre.pro (talks only via the DB), so
the app stays proprietary — see README. We run zero Ultralytics code (pure onnxruntime).

CHANNELS: unlike the old ifnude path, v3's `detect()` reads a FILE PATH via cv2.imread internally,
so there is no RGB/BGR ambiguity — we just write the PIL image to a temp file and hand over the path.
"""
import os
import tempfile
import threading

from PIL import Image
from nudenet import NudeDetector

# All 18 classes the v3 detector can output (drives the tester board). The "*_EXPOSED" ones gate.
LABELS = [
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED",
    "FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED", "BUTTOCKS_EXPOSED",
    "ARMPITS_EXPOSED", "FEET_EXPOSED", "BELLY_EXPOSED",
    "FEMALE_GENITALIA_COVERED", "FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED",
    "ANUS_COVERED", "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED",
    "FACE_FEMALE", "FACE_MALE",
]

# v3 keeps every detection scoring >= 0.2 (hardcoded in its postprocess) — so raw scores down to 20%
# surface and OUR config.verdict thresholds do the gating. Resolution 320 = bundled 320n weights;
# set NSFW_INFER_RES=640 + NSFW_MODEL_PATH=<640m.onnx> for higher accuracy.
_INFER_RES = int(os.environ.get("NSFW_INFER_RES", "320"))
_MODEL_PATH = os.environ.get("NSFW_MODEL_PATH") or None  # None → the bundled 320n.onnx

_init_lock = threading.Lock()
_detector = None


def _det():
    global _detector
    if _detector is None:
        with _init_lock:
            if _detector is None:
                _detector = NudeDetector(model_path=_MODEL_PATH, inference_resolution=_INFER_RES)
    return _detector


def _detect_path(path: str) -> dict:
    """One file path → {class: max_confidence} across all boxes (detected classes only)."""
    dets = _det().detect(path) or []
    out = {}
    for d in dets:
        lbl = d.get("class")
        sc = float(d.get("score", 0.0))
        if lbl and sc > out.get(lbl, 0.0):
            out[lbl] = sc
    return out


def detect_image(img) -> dict:
    """One image (PIL.Image OR an existing file path) → {class: max_confidence}."""
    if isinstance(img, str):
        return _detect_path(img)
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        img.convert("RGB").save(path, "JPEG", quality=95)
        return _detect_path(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def detect_max(images) -> dict:
    """Aggregate per-class MAX across many images/frames → {class: max_confidence}."""
    agg = {}
    for im in images:
        for lbl, sc in detect_image(im).items():
            if sc > agg.get(lbl, 0.0):
                agg[lbl] = sc
    return agg


def warmup():
    try:
        detect_image(Image.new("RGB", (320, 320), (127, 127, 127)))  # loads the ONNX session
    except Exception as e:
        print(f"[nsfw] warmup skipped: {type(e).__name__}: {e}", flush=True)
