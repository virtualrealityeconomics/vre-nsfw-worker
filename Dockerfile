FROM python:3.11-slim
# ffmpeg = video frame sampling. opencv-python-headless (pulled by nudenet) needs no GUI libs, but
# libgl1/libglib2 are kept as a cheap safety net for onnxruntime/opencv on slim.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# The NudeNet v3 model (320n.onnx) ships inside the nudenet wheel — nothing to download. Warm the
# ONNX session once at build so first-request latency is low and the import is proven at build time.
RUN python -c "from nudenet import NudeDetector; NudeDetector(); print('nudenet v3 ready')"
COPY . .
CMD ["python", "main.py"]
