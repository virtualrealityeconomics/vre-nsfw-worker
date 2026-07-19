# ── Stage 1: export Falconsai/nsfw_image_detection → ONNX (heavy build-only deps live here) ──────
FROM python:3.11-slim AS builder
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "optimum[exporters]" transformers onnx
# Downloads the model from HuggingFace and writes /export/model.onnx (input: pixel_values 1x3x224x224).
RUN optimum-cli export onnx --model Falconsai/nsfw_image_detection --task image-classification /export

# ── Stage 2: lean runtime (onnxruntime + ffmpeg + the baked model only — no torch) ───────────────
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=builder /export/model.onnx /models/nsfw.onnx
COPY . .
# Headless poller; the only listener is the secret-gated /metrics on $PORT.
CMD ["python", "main.py"]
