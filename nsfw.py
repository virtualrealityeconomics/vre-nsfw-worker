"""Falconsai/nsfw_image_detection via ONNX Runtime (CPU).

Falconsai is a ViT-base-patch16-224 binary classifier. Its image processor: resize to 224x224,
rescale 1/255, normalize with mean/std = 0.5 (per-channel). Output = logits[2] for labels
['normal', 'nsfw']; softmax → P(nsfw) = probs[1]. Lazy-loaded; batches frames in one run.
"""
import threading

import numpy as np
import onnxruntime as ort
from PIL import Image

import config

_lock = threading.Lock()
_session = None
_input_name = None

_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
_SIZE = 224


def _load():
    global _session, _input_name
    if _session is None:
        with _lock:
            if _session is None:
                so = ort.SessionOptions()
                so.intra_op_num_threads = 0  # let ORT pick; CPU box
                _session = ort.InferenceSession(config.MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
                _input_name = _session.get_inputs()[0].name
    return _session, _input_name


def _preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((_SIZE, _SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC, 0..1
    arr = np.transpose(arr, (2, 0, 1))                        # CHW
    arr = (arr - _MEAN) / _STD                                # normalize
    return arr


def _softmax_nsfw(logits: np.ndarray) -> np.ndarray:
    # logits: (N, 2) → P(nsfw) per row (label index 1)
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    probs = e / e.sum(axis=1, keepdims=True)
    return probs[:, 1]


def score_images(images, batch_size: int = 16):
    """images: iterable of PIL.Image → np.ndarray of P(nsfw), one per image. Empty in → empty out."""
    imgs = list(images)
    if not imgs:
        return np.array([], dtype=np.float32)
    session, input_name = _load()
    out = []
    for i in range(0, len(imgs), batch_size):
        batch = np.stack([_preprocess(im) for im in imgs[i:i + batch_size]]).astype(np.float32)
        logits = session.run(None, {input_name: batch})[0]
        out.append(_softmax_nsfw(np.asarray(logits, dtype=np.float32)))
    return np.concatenate(out)


def warmup():
    """Load the model + one dummy inference so the first real scan isn't a cold start."""
    try:
        score_images([Image.new("RGB", (_SIZE, _SIZE), (127, 127, 127))])
    except Exception as e:
        print(f"[nsfw] warmup skipped: {type(e).__name__}: {e}", flush=True)
