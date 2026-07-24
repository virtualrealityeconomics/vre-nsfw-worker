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
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

import config
import db
import frames
import gate
import nsfw
import r2
import sysmetrics

_stop = threading.Event()

# ── Logging: human Israel-time stamp + "what am I testing" context. tzdata may be absent on a slim
# image → fall back to UTC rather than crash the whole worker over a cosmetic timestamp.
try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo(os.environ.get("NSFW_LOG_TZ", "Asia/Jerusalem"))
except Exception:
    _TZ = timezone.utc


def _log(msg):
    print(f"[{datetime.now(_TZ).strftime('%Y-%m-%d %H:%M:%S')} IL] {msg}", flush=True)


def _fname(url):
    """Uploaded filename from an R2 url — so logs say WHAT is being scanned."""
    return ((url or "").split("?")[0].rsplit("/", 1)[-1] or "?")[:60]


def _top(scores):
    """Compact 'label=score' of the strongest detection, for logs."""
    if not scores:
        return "clean"
    lbl = max(scores, key=scores.get)
    return f"{lbl}={round(scores[lbl], 3)}"


# ── Image (optimistic: already public, so we only ACT on flagged/rejected) ───────────────────────
def process_image_post(row):
    t0 = time.time()
    urls = [u for u in (row.get("mediaUrls") or []) if u]
    imgs = []
    for u in urls:
        try:
            data = r2.fetch_public(u, max_bytes=config.MAX_IMAGE_BYTES)
            imgs.append(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception as e:
            _log(f"[image] fetch failed post={row['id']}: {type(e).__name__}: {e}")
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
    ctx = db.describe(row["id"])
    _log(f'[image] "{ctx["title"]}" by {ctx["author"]} · {_fname(urls[0] if urls else "")} · '
         f'{_top(d.get("scores") or {})}[{d.get("layer")}:{d.get("reason","")}] '
         f'-> {status.upper()} {"✅" if status == "approved" else "❌"} · {time.time()-t0:.1f}s')


# ── Video ─────────────────────────────────────────────────────────────────────────────────────────
def _scan_video(url):
    """Download → sample frames → per-frame NudeNet. Returns (agg, suspicious):
        agg        = {label: max across all frames}   (for logs / persistence)
        suspicious = [(frame_img, frame_scores)] for frames NudeNet didn't clear (kept for the vision pass).
    None = unscannable → retry (NEVER approve unseen)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
        r2.download_to(url, tf.name)
        fr = frames.sample(tf.name)
    if not fr:
        return None
    agg, suspicious = {}, []
    for im in fr:
        sc = nsfw.detect_image(im)
        for lbl, v in sc.items():
            if v > agg.get(lbl, 0.0):
                agg[lbl] = v
        if config.verdict(sc)[0] != "approved":
            suspicious.append((im, sc))
    return agg, suspicious, len(fr)


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
            _log(f"[thumb] fetch failed post={post_id}: {type(e).__name__}: {e}")
    if not imgs:
        return "approved", {}, urls
    # free_block=False: the poster is a video frame too — let vision rescue a squirrel-belly poster,
    # while explicit nudity still hard-blocks (decide_image step 1).
    d = gate.decide_images(imgs, free_block=False)
    return d["status"], d.get("scores") or {}, urls


def process_video(row):
    t0 = time.time()
    _log(f'[video] ▶ NEW {_fname(row["videoUrl"])} — downloading + analyzing… (id={row["id"]})')
    # Whole critical path (scan → decide → thumbnail → resolve) is wrapped: any error → db.fail (bounded
    # retries → terminal 'error'), never a silently-stuck row. Quarantine + log run AFTER, best-effort.
    try:
        res = _scan_video(row["videoUrl"])
        if res is None:
            db.fail("Video", row["id"], row["moderationAttempts"])
            return
        agg, suspicious, n = res
        d = gate.decide_video(agg, suspicious)                    # clean → free; elegance band → vision
        t_status, t_scores, _media = _scan_thumbnail(row.get("postId"))   # CR2
        status = _worst(d["status"], t_status)
        labels = dict(agg); labels["_reason"] = d.get("reason", ""); labels["_layer"] = d.get("layer", "")
        labels.update({"thumb_" + k: v for k, v in t_scores.items()})
        db.resolve_video(row["id"], row.get("postId"), status, round(d.get("score", 0.0), 4), labels)
    except Exception as e:
        _log(f"[video] scan failed id={row['id']}: {type(e).__name__}: {e}")
        db.fail("Video", row["id"], row["moderationAttempts"])
        return
    if status == "rejected":
        thumb_keys = [r2.url_to_key(u) for u in (db.post_media_all_urls(row.get("postId")) or [])]
        r2.quarantine_keys(_video_keys(row) + thumb_keys)
    ctx = db.describe(row.get("postId"))
    _log(f'[video] "{ctx["title"]}" by {ctx["author"]} · {_fname(row["videoUrl"])} · {n}f/{len(suspicious)}susp · '
         f'v={_top(agg)}[{d.get("layer")}:{d.get("reason","")}] vision={d.get("vision_frames",0)} thumb={t_status} '
         f'-> {status.upper()} {"✅" if status == "approved" else "❌"} · {time.time()-t0:.1f}s')


def process_orphan_video_post(row):
    t0 = time.time()
    _log(f'[orphan] ▶ NEW {_fname(row["videoUrl"])} — downloading + analyzing… (post={row["id"]})')
    try:
        res = _scan_video(row["videoUrl"])
        if res is None:
            db.fail("Post", row["id"], row["moderationAttempts"])
            return
        agg, suspicious, n = res
        d = gate.decide_video(agg, suspicious)   # SFW → 'approved' publishes the raw MP4 (no transcode)
        t_status, t_scores, _media = _scan_thumbnail(row["id"])   # CR2
        status = _worst(d["status"], t_status)
        labels = dict(agg); labels["_reason"] = d.get("reason", ""); labels["_layer"] = d.get("layer", "")
        labels.update({"thumb_" + k: v for k, v in t_scores.items()})
        db.resolve_post(row["id"], status, round(d.get("score", 0.0), 4), labels)
    except Exception as e:
        _log(f"[orphan] scan failed post={row['id']}: {type(e).__name__}: {e}")
        db.fail("Post", row["id"], row["moderationAttempts"])
        return
    if status == "rejected":
        thumb_keys = [r2.url_to_key(u) for u in (db.post_media_all_urls(row["id"]) or [])]
        r2.quarantine_keys([r2.url_to_key(row.get("videoUrl"))] + thumb_keys)
    ctx = db.describe(row["id"])
    _log(f'[orphan] "{ctx["title"]}" by {ctx["author"]} · {_fname(row["videoUrl"])} · {n}f/{len(suspicious)}susp · '
         f'v={_top(agg)}[{d.get("layer")}:{d.get("reason","")}] vision={d.get("vision_frames",0)} thumb={t_status} '
         f'-> {status.upper()} {"✅" if status == "approved" else "❌"} · {time.time()-t0:.1f}s')


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
# Bind AF_INET / 0.0.0.0 so the admin dashboard (Railway) can poll this box on its public IPv4, exactly
# like vre-video-worker. The old IPv6-only (`::`) bind was a Railway-internal assumption that never
# applied here — this worker runs on Vultr — and it REFUSED the dashboard's IPv4 connection.
class _MetricsServer(ThreadingHTTPServer):
    daemon_threads = True


class _Metrics(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] != "/metrics":
            self.send_response(404); self.end_headers(); return
        if config.METRICS_SECRET and self.headers.get("x-metrics-secret") != config.METRICS_SECRET:
            self.send_response(401); self.end_headers(); return
        try:
            # System stats first (pure-local, always works) so the Servers card renders even if the DB
            # is momentarily down; the DB-backed moderation queue degrades to {error} on its own.
            payload = sysmetrics.collect()
            try:
                dbm = db.metrics()  # {post, video, backlog}
                bk = dbm.get("backlog", {}) or {}
                payload["queue"] = {
                    "pending": (bk.get("images") or 0) + (bk.get("videos") or 0),  # the scale signal
                    "processing": 0,  # NudeNet is per-row + fast; no distinct in-flight status to report
                    "failed": (dbm.get("post", {}).get("error") or 0) + (dbm.get("video", {}).get("error") or 0),
                }
                payload["moderation"] = dbm  # full status breakdown for anyone who wants the detail
            except Exception as e:
                payload["queue"] = {"error": str(e)}
            body = json.dumps(payload).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(500); self.end_headers(); self.wfile.write(str(e).encode())

    def log_message(self, *a):
        pass


def _serve_metrics():
    try:
        _MetricsServer(("0.0.0.0", config.METRICS_PORT), _Metrics).serve_forever()
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
