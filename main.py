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

from PIL import Image

import config
import db
import frames
import gate
import nsfw
import r2

_stop = threading.Event()


def _top(scores):
    """Compact 'label=score' of the strongest detection, for logs."""
    if not scores:
        return "clean"
    lbl = max(scores, key=scores.get)
    return f"{lbl}={round(scores[lbl], 3)}"


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
    if not imgs:
        db.fail("Post", row["id"], row["moderationAttempts"])
        return
    d = gate.decide_images(imgs)                       # NudeNet free pre-block → Claude vision
    status = d["status"]
    labels = dict(d.get("scores") or {})               # persist the reason + which layer decided
    labels["_reason"] = d.get("reason", "")
    labels["_layer"] = d.get("layer", "")
    db.resolve_post(row["id"], status, round(d.get("score", 0.0), 4), labels)
    if status == "rejected":
        keys = [r2.url_to_key(u) for u in (row.get("mediaUrls") or [])
                + (row.get("mediaUrlsSmall") or []) + (row.get("mediaUrlsMedium") or [])]
        r2.quarantine_keys(keys)
    print(f"[image] post={row['id']} {_top(d.get('scores') or {})} [{d.get('layer')}] -> {status} ({d.get('reason','')})", flush=True)


# ── Video ─────────────────────────────────────────────────────────────────────────────────────────
def _scan_video(url):
    """Download → sample frames → per-label MAX scores. None = unscannable → retry (NEVER approve unseen)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
        r2.download_to(url, tf.name)
        fr = frames.sample(tf.name)
    if not fr:
        return None
    return nsfw.detect_max(fr)


def _video_keys(row):
    keys = [r2.url_to_key(row.get(k)) for k in ("videoUrl", "previewUrl", "thumbnailUrl", "thumbnailSmallUrl", "hlsUrl")]
    hls = r2.url_to_key(row.get("hlsUrl"))
    if hls and "/" in hls:  # sweep the whole HLS dir (variable .ts segment count)
        keys += r2.list_prefix(hls.rsplit("/", 1)[0] + "/")
    return keys


_RANK = {"approved": 0, "flagged": 1, "rejected": 2, "error": 3}


def _worst(a, b):
    return a if _RANK.get(a, 0) >= _RANK.get(b, 0) else b


def _scan_thumbnail(post_id):
    """CR2: a video post's client thumbnail (Post.mediaUrls) renders as the public poster but is
    scanned by NO image loop (claim_image_post excludes posts with a videoUrl). Scan it here with the
    full image gate so an SFW video with an NSFW thumbnail can't slip through. → (status, scores)."""
    urls = [u for u in (db.post_media_urls(post_id) or []) if u]
    imgs = []
    for u in urls:
        try:
            data = r2.fetch_public(u, max_bytes=config.MAX_IMAGE_BYTES)
            imgs.append(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception as e:
            print(f"[thumb] fetch failed post={post_id}: {type(e).__name__}: {e}", flush=True)
    if not imgs:
        return "approved", {}, urls
    d = gate.decide_images(imgs)
    return d["status"], d.get("scores") or {}, urls


def process_video(row):
    try:
        scores = _scan_video(row["videoUrl"])
    except Exception as e:
        print(f"[video] scan failed id={row['id']}: {type(e).__name__}: {e}", flush=True)
        db.fail("Video", row["id"], row["moderationAttempts"])
        return
    if scores is None:
        db.fail("Video", row["id"], row["moderationAttempts"])
        return
    status, _, sc = config.verdict(scores)
    t_status, t_scores, _media = _scan_thumbnail(row.get("postId"))  # CR2 (quarantine set re-derived below)
    status = _worst(status, t_status)
    labels = dict(scores)
    labels.update({"thumb_" + k: v for k, v in t_scores.items()})
    db.resolve_video(row["id"], row.get("postId"), status, round(sc, 4), labels)
    if status == "rejected":
        # R2: quarantine the FULL thumbnail derivative set (Small/Medium too), not just what we scanned.
        thumb_keys = [r2.url_to_key(u) for u in (db.post_media_all_urls(row.get("postId")) or [])]
        r2.quarantine_keys(_video_keys(row) + thumb_keys)
    print(f"[video] id={row['id']} v={_top(scores)} thumb={t_status} -> {status}", flush=True)


def process_orphan_video_post(row):
    try:
        scores = _scan_video(row["videoUrl"])
    except Exception as e:
        print(f"[orphan] scan failed post={row['id']}: {type(e).__name__}: {e}", flush=True)
        db.fail("Post", row["id"], row["moderationAttempts"])
        return
    if scores is None:
        db.fail("Post", row["id"], row["moderationAttempts"])
        return
    status, _, sc = config.verdict(scores)  # SFW → 'approved' publishes it (raw MP4; no Video to transcode)
    t_status, t_scores, _media = _scan_thumbnail(row["id"])  # CR2 (quarantine set re-derived below)
    status = _worst(status, t_status)
    labels = dict(scores)
    labels.update({"thumb_" + k: v for k, v in t_scores.items()})
    db.resolve_post(row["id"], status, round(sc, 4), labels)
    if status == "rejected":
        # R2: quarantine the FULL thumbnail derivative set (Small/Medium too), not just what we scanned.
        thumb_keys = [r2.url_to_key(u) for u in (db.post_media_all_urls(row["id"]) or [])]
        r2.quarantine_keys([r2.url_to_key(row.get("videoUrl"))] + thumb_keys)
    print(f"[orphan] post={row['id']} v={_top(scores)} thumb={t_status} -> {status}", flush=True)


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
    print(f"[boot] nsfw-worker (NudeNet v3); block-thresholds={config.BLOCK_THRESHOLDS}", flush=True)
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
