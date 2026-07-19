"""vre-nsfw-worker — headless NSFW moderation poller.

Two independent threads so a slow video scan can NEVER stall image publishes:
  • image loop  (fast poll) — optimistic images already public; retro-hide + quarantine if NSFW.
  • video loop  (slower)    — pending videos + orphan video-posts; SFW → hand to the Vultr transcode
                              worker (which claims only moderationStatus='approved'); NSFW → hide+quarantine.

Scale horizontally: run N replicas — claims are lease + FOR UPDATE SKIP LOCKED, so each grabs
different rows and a crash self-heals (stale lease → re-claimable). No coordinator needed.
"""
import io
import json
import socket
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

import config
import db
import frames
import nsfw
import r2

_stop = threading.Event()


def _verdict(p_nsfw):
    """3-band gate on the aggregated P(nsfw)."""
    if p_nsfw >= config.T_BLOCK:
        return "rejected"      # hide + quarantine
    if p_nsfw >= config.T_PASS:
        return "flagged"       # hide; admin reviews
    return "approved"


def _aggregate(scores):
    """MAX (or a high percentile) — one explicit frame/image fails the whole item."""
    if scores is None or len(scores) == 0:
        return None
    a = np.asarray(scores, dtype=np.float32)
    if config.VIDEO_AGG_PCT >= 1.0:
        return float(a.max())
    return float(np.percentile(a, config.VIDEO_AGG_PCT * 100))


# ── Image (optimistic: already public, so we only ACT on flagged/rejected) ───────────────────────
def process_image_post(row):
    urls = [u for u in (row.get("mediaUrls") or []) if u]
    imgs = []
    for u in urls:
        try:
            data = r2.fetch_public(u, max_bytes=config.MAX_IMAGE_BYTES)
            imgs.append(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception as e:
            print(f"[image] fetch failed post={row['id']}: {type(e).__name__}: {e}", flush=True)
            db.fail("Post", row["id"], row["moderationAttempts"])
            return
    p = _aggregate(nsfw.score_images(imgs))
    if p is None:
        db.fail("Post", row["id"], row["moderationAttempts"])
        return
    status = _verdict(p)
    db.resolve_post(row["id"], status, round(p, 4), {"pnsfw": round(p, 4), "count": len(urls)})
    if status == "rejected":
        keys = [r2.url_to_key(u) for u in (row.get("mediaUrls") or [])
                + (row.get("mediaUrlsSmall") or []) + (row.get("mediaUrlsMedium") or [])]
        r2.quarantine_keys(keys)
    print(f"[image] post={row['id']} p={round(p, 3)} -> {status}", flush=True)


# ── Video ─────────────────────────────────────────────────────────────────────────────────────────
def _scan_video(url):
    """Download → sample frames → aggregated P(nsfw). None = unscannable → retry (NEVER approve unseen)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
        r2.download_to(url, tf.name)
        fr = frames.sample(tf.name)
    if not fr:
        return None
    return _aggregate(nsfw.score_images(fr))


def _video_keys(row):
    keys = [r2.url_to_key(row.get(k)) for k in ("videoUrl", "previewUrl", "thumbnailUrl", "thumbnailSmallUrl", "hlsUrl")]
    hls = r2.url_to_key(row.get("hlsUrl"))
    if hls and "/" in hls:  # sweep the whole HLS dir (variable .ts segment count)
        keys += r2.list_prefix(hls.rsplit("/", 1)[0] + "/")
    return keys


def process_video(row):
    try:
        p = _scan_video(row["videoUrl"])
    except Exception as e:
        print(f"[video] scan failed id={row['id']}: {type(e).__name__}: {e}", flush=True)
        db.fail("Video", row["id"], row["moderationAttempts"])
        return
    if p is None:
        db.fail("Video", row["id"], row["moderationAttempts"])
        return
    status = _verdict(p)
    db.resolve_video(row["id"], row.get("postId"), status, round(p, 4), {"pnsfw": round(p, 4)})
    if status == "rejected":
        r2.quarantine_keys(_video_keys(row))
    print(f"[video] id={row['id']} p={round(p, 3)} -> {status}", flush=True)


def process_orphan_video_post(row):
    try:
        p = _scan_video(row["videoUrl"])
    except Exception as e:
        print(f"[orphan] scan failed post={row['id']}: {type(e).__name__}: {e}", flush=True)
        db.fail("Post", row["id"], row["moderationAttempts"])
        return
    if p is None:
        db.fail("Post", row["id"], row["moderationAttempts"])
        return
    status = _verdict(p)  # SFW → 'approved' publishes it (plays raw MP4; there's no Video to transcode)
    db.resolve_post(row["id"], status, round(p, 4), {"pnsfw": round(p, 4)})
    if status == "rejected":
        r2.quarantine_keys([r2.url_to_key(row.get("videoUrl"))])
    print(f"[orphan] post={row['id']} p={round(p, 3)} -> {status}", flush=True)


# ── Loops ─────────────────────────────────────────────────────────────────────────────────────────
def image_loop():
    while not _stop.is_set():
        try:
            row = db.claim_image_post()
            if row:
                process_image_post(row)
                continue
        except Exception:
            traceback.print_exc()
        _stop.wait(config.IMAGE_POLL_SEC)


def video_loop():
    while not _stop.is_set():
        try:
            row = db.claim_video()
            if row:
                process_video(row)
                continue
            row = db.claim_orphan_video_post()
            if row:
                process_orphan_video_post(row)
                continue
        except Exception:
            traceback.print_exc()
        _stop.wait(config.VIDEO_POLL_SEC)


# ── Metrics (secret-gated; a backed-up/down worker = a growing backlog the dash can alert on) ─────
class _V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6  # Railway internal networking is IPv6
    daemon_threads = True


class _Metrics(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] != "/metrics":
            self.send_response(404); self.end_headers(); return
        if config.METRICS_SECRET and self.headers.get("x-metrics-secret") != config.METRICS_SECRET:
            self.send_response(401); self.end_headers(); return
        try:
            body = json.dumps(db.metrics()).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(500); self.end_headers(); self.wfile.write(str(e).encode())

    def log_message(self, *a):
        pass


def _serve_metrics():
    try:
        _V6Server(("::", config.METRICS_PORT), _Metrics).serve_forever()
    except Exception as e:
        print(f"[metrics] server failed: {type(e).__name__}: {e}", flush=True)


def main():
    print(f"[boot] nsfw-worker starting (T_PASS={config.T_PASS} T_BLOCK={config.T_BLOCK})", flush=True)
    if not config.DIRECT_URL:
        print("[boot] FATAL: DIRECT_URL unset", flush=True)
        sys.exit(1)
    nsfw.warmup()
    threading.Thread(target=_serve_metrics, daemon=True).start()
    threading.Thread(target=image_loop, daemon=True).start()
    threading.Thread(target=video_loop, daemon=True).start()
    print(f"[boot] loops running; /metrics on :{config.METRICS_PORT}", flush=True)
    try:
        while not _stop.is_set():
            time.sleep(3600)
    except KeyboardInterrupt:
        _stop.set()


if __name__ == "__main__":
    main()
