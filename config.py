"""Configuration for vre-nsfw-worker — the NSFW moderation poller.

Headless: reads Postgres (DIRECT_URL), fetches media from the PUBLIC R2 URL, classifies with
NudeNet v3 (ONNX, per-body-part), writes moderationStatus. No inbound HTTP except a secret-gated
/metrics. All tunables are env so thresholds can be calibrated on real labels without a redeploy.
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

# ── Model: NudeNet v3 via the MIT `nudenet` package (320n.onnx bundled in the wheel; onnxruntime) ─

# ── ELEGANCE gate — VRE is a LinkedIn-style ELEGANT/professional platform ───────────────────────
#   We reject not only NUDITY but also "covered-but-revealing" content — bikinis, lingerie,
#   underwear, bare midriff, shirtless. NudeNet gives per-body-part boxes; we treat a chosen set of
#   them as REJECT signals (a body-part proxy for "not elegant").
#     REJECT (hidden + quarantined) if ANY block-label ≥ its threshold.
#     FLAG   (hidden; admin reviews) if ANY flag-label  ≥ its threshold (the review band just under).
#     else APPROVE (elegant / professionally clothed).
#   NOTE: tight crops that show NO gated part (e.g. a smiling face over a bikini top) can still slip
#   through — that residual needs the vision-LLM layer. All thresholds are env-tunable.
def _f(name, default):
    return float(os.environ.get(name, default))


# AUTO-REJECT — high-precision signals only. In testing these fire on revealing content but NOT on
# elegant/professionally-dressed people, so blocking on them does not kill legit photos.
BLOCK_THRESHOLDS = {
    # explicit nudity
    "FEMALE_GENITALIA_EXPOSED": _f("NSFW_BLOCK_GENITALIA_F", "0.30"),
    "MALE_GENITALIA_EXPOSED":   _f("NSFW_BLOCK_GENITALIA_M", "0.30"),
    "ANUS_EXPOSED":             _f("NSFW_BLOCK_ANUS", "0.30"),
    "FEMALE_BREAST_EXPOSED":    _f("NSFW_BLOCK_BREAST_F", "0.30"),
    "BUTTOCKS_EXPOSED":         _f("NSFW_BLOCK_BUTTOCKS", "0.35"),
    "MALE_BREAST_EXPOSED":      _f("NSFW_BLOCK_BREAST_M", "0.50"),  # shirtless man
    # clean "not-elegant" separators (fire on bikini/thong/bare-midriff, NOT on elegant attire)
    "BELLY_EXPOSED":            _f("NSFW_BLOCK_BELLY", "0.50"),            # bare midriff
    "FEMALE_GENITALIA_COVERED": _f("NSFW_BLOCK_GENITALIA_F_COV", "0.45"),  # thong / bikini bottom
}
# HUMAN-REVIEW (FLAG, hidden until an admin decides) — AMBIGUOUS signals whose scores OVERLAP
# between elegant clothing and cleavage/lingerie/bikini-tops, so auto-rejecting them would also
# reject legit elegant photos. We can't separate these with body-parts alone → send to review.
# (Set NSFW_STRICT_COVERED=1 to instead AUTO-REJECT the breast/buttocks-covered band — stricter,
# but it WILL false-reject some elegant photos.)
FLAG_THRESHOLDS = {
    "FEMALE_BREAST_COVERED":    _f("NSFW_FLAG_BREAST_F_COV", "0.60"),  # cleavage / bikini top / bra
    "BUTTOCKS_COVERED":         _f("NSFW_FLAG_BUTTOCKS_COV", "0.60"),  # tight / revealing bottom
    # low-confidence nudity → review rather than silently approve
    "FEMALE_GENITALIA_EXPOSED": _f("NSFW_FLAG_GENITALIA_F", "0.20"),
    "MALE_GENITALIA_EXPOSED":   _f("NSFW_FLAG_GENITALIA_M", "0.20"),
    "ANUS_EXPOSED":             _f("NSFW_FLAG_ANUS", "0.20"),
    "FEMALE_BREAST_EXPOSED":    _f("NSFW_FLAG_BREAST_F", "0.20"),
    "BUTTOCKS_EXPOSED":         _f("NSFW_FLAG_BUTTOCKS", "0.23"),
    "MALE_BREAST_EXPOSED":      _f("NSFW_FLAG_BREAST_M", "0.35"),
    "BELLY_EXPOSED":            _f("NSFW_FLAG_BELLY", "0.38"),
}
# Strict mode: promote the ambiguous covered-band into AUTO-REJECT (kills more bikinis/lingerie, at
# the cost of false-rejecting some elegant photos — your call, env-toggled).
if os.environ.get("NSFW_STRICT_COVERED", "") in ("1", "true", "yes"):
    BLOCK_THRESHOLDS["FEMALE_BREAST_COVERED"] = FLAG_THRESHOLDS.pop("FEMALE_BREAST_COVERED")
    BLOCK_THRESHOLDS["BUTTOCKS_COVERED"] = FLAG_THRESHOLDS.pop("BUTTOCKS_COVERED")


def verdict(scores):
    """scores = {label: confidence}. → (status, reason_label, reason_score). status ∈ approved|flagged|rejected."""
    hits = [(l, scores[l]) for l, t in BLOCK_THRESHOLDS.items() if scores.get(l, 0.0) >= t]
    if hits:
        l, s = max(hits, key=lambda x: x[1]);  return "rejected", l, s
    hits = [(l, scores[l]) for l, t in FLAG_THRESHOLDS.items() if scores.get(l, 0.0) >= t]
    if hits:
        l, s = max(hits, key=lambda x: x[1]);  return "flagged", l, s
    return "approved", None, (max(scores.values()) if scores else 0.0)

# ── Claude vision (hybrid layer 2 — IMAGES only; videos stay NudeNet-only to avoid per-frame cost) ─
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VISION_ENABLED    = os.environ.get("NSFW_VISION_ENABLED", "1") not in ("0", "false", "no")
# On a vision API error: default holds the image for review (safe); fail-open trusts the free gate.
VISION_FAIL_OPEN  = os.environ.get("NSFW_VISION_FAIL_OPEN", "") in ("1", "true", "yes")

# ── Video frame sampling (ffmpeg) ──────────────────────────────────────────────────────────────
SCENE_THRESHOLD = float(os.environ.get("NSFW_SCENE_THRESHOLD", "0.3"))  # select='gt(scene,X)'
FPS_FLOOR       = float(os.environ.get("NSFW_FPS_FLOOR", "0.5"))         # even-sample rate target
MIN_FRAMES      = int(os.environ.get("NSFW_MIN_FRAMES", "16"))           # guaranteed floor per video
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
