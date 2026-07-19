"""Configuration for vre-nsfw-worker — the NSFW moderation poller.

Headless: reads Postgres (DIRECT_URL), fetches media from the PUBLIC R2 URL, classifies with
Falconsai (ONNX), writes moderationStatus. No inbound HTTP except a secret-gated /metrics.
All tunables are env so thresholds can be calibrated on real labels without a redeploy.
"""
import os

# ── Database (poll the same Postgres the app uses; DIRECT connection, not the pooler) ─────────
DIRECT_URL = os.environ.get("DIRECT_URL", "")

# ── R2 (Cloudflare) — public reads via media.vre.pro; creds only needed to QUARANTINE on block ─
R2_ENDPOINT       = os.environ.get("R2_ENDPOINT", "")
R2_ACCESS_KEY_ID  = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME    = os.environ.get("R2_BUCKET_NAME", "")
MEDIA_BASE        = os.environ.get("MEDIA_BASE", "https://media.vre.pro/")  # public URL prefix → key
QUARANTINE_PREFIX = os.environ.get("QUARANTINE_PREFIX", "quarantine/")       # blocked bytes move HERE (kept, not deleted)

# ── Model ─────────────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("NSFW_MODEL_PATH", "/models/nsfw.onnx")  # baked into the image at build

# ── 3-band gate on P(nsfw) (calibrate on our own labels; env-tunable) ──────────────────────────
#   P <  T_PASS            → approved
#   T_PASS ≤ P <  T_BLOCK  → flagged  (hidden; admin reviews)
#   P ≥  T_BLOCK           → rejected (hidden + bytes moved to the quarantine prefix so the URL 404s)
T_PASS  = float(os.environ.get("NSFW_T_PASS", "0.5"))
T_BLOCK = float(os.environ.get("NSFW_T_BLOCK", "0.85"))

# ── Video frame sampling (ffmpeg) ──────────────────────────────────────────────────────────────
SCENE_THRESHOLD = float(os.environ.get("NSFW_SCENE_THRESHOLD", "0.3"))  # select='gt(scene,X)'
FPS_FLOOR       = float(os.environ.get("NSFW_FPS_FLOOR", "0.5"))         # uniform floor sample
MAX_FRAMES      = int(os.environ.get("NSFW_MAX_FRAMES", "150"))          # cap per video
VIDEO_AGG_PCT   = float(os.environ.get("NSFW_VIDEO_AGG_PCT", "1.0"))     # 1.0 = MAX; 0.95 = 95th pct

# ── Reliability (audit C1 — never leave a legit upload invisibly stuck) ─────────────────────────
MAX_ATTEMPTS   = int(os.environ.get("NSFW_MAX_ATTEMPTS", "3"))   # then a terminal 'error' state
LEASE_MINUTES  = int(os.environ.get("NSFW_LEASE_MINUTES", "10")) # 'processing' older than this → reclaimed to 'pending'

# ── Poll cadence (image loop fast so a long video scan can't starve image publishes) ────────────
IMAGE_POLL_SEC = float(os.environ.get("NSFW_IMAGE_POLL_SEC", "2.5"))
VIDEO_POLL_SEC = float(os.environ.get("NSFW_VIDEO_POLL_SEC", "5"))
RECLAIM_SEC    = float(os.environ.get("NSFW_RECLAIM_SEC", "60"))

# ── Metrics (secret-gated, mirrors vre-video-worker) ───────────────────────────────────────────
METRICS_PORT   = int(os.environ.get("PORT", os.environ.get("METRICS_PORT", "8787")))
METRICS_SECRET = os.environ.get("METRICS_SECRET", "")

MAX_IMAGE_BYTES = int(os.environ.get("NSFW_MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))  # decode-DoS guard
