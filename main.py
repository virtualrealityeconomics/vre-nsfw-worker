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
def _aggregate(per_frame, pct):
    """Combine per-frame scores into one score per label.

    pct=1.0 is MAX (any single frame decides). Anything lower takes that percentile across frames.

    WHY THIS IS CONFIGURABLE, AND WHY MAX IS DANGEROUS ON LONG VIDEOS:
    frames.sample() returns up to MAX_FRAMES (150) frames. Taking the MAX over 150 samples of a
    detector with even a 1% per-frame false-positive rate rejects an ordinary video 78% of the time
    (1 - 0.99**150) — the failure is arithmetic, not bad luck. Observed live: an 8-minute video of
    GitHub star-history CHARTS was rejected on a single frame scoring 0.3414 for
    MALE_GENITALIA_EXPOSED against a 0.30 threshold.

    config.VIDEO_AGG_PCT has existed (and been documented in .env.example) since the worker was
    written, but NOTHING EVER READ IT — this function is what makes the knob real. Setting it before
    would have looked like a fix and changed nothing.
    """
    out = {}
    for lbl, vals in per_frame.items():
        if not vals:
            continue
        if pct >= 1.0 or len(vals) == 1:
            out[lbl] = max(vals)
        else:
            s = sorted(vals)
            # Nearest-rank percentile; index clamped so pct<1 always drops at least the top sample
            # (otherwise a 150-frame video and a 3-frame video behave completely differently).
            idx = min(len(s) - 1, max(0, int(round(pct * len(s))) - 1))
            out[lbl] = s[idx]
    return out


def _scan_video(url):
    """Download → sample frames → per-frame NudeNet. Returns (agg, suspicious):
        agg        = {label: aggregated across frames per config.VIDEO_AGG_PCT}
        suspicious = [(frame_img, frame_scores)] for frames NudeNet didn't clear (kept for the vision pass).
    None = unscannable → retry (NEVER approve unseen)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
        r2.download_to(url, tf.name)
        fr, complete = frames.sample(tf.name)
    if not fr:
        return None
    per_frame, suspicious = {}, []
    for im in fr:
        sc = nsfw.detect_image(im)
        for lbl, v in sc.items():
            per_frame.setdefault(lbl, []).append(v)
        # video=True so "suspicious" is decided on the same band decide_video will judge it with.
        # Mixing the two means frames that are perfectly fine under the video band still get carried
        # into the decision path.
        if config.verdict(sc, video=True)[0] != "approved":
            suspicious.append((im, sc))
    # `suspicious` is deliberately still built per-frame from the RAW scores: aggregation decides
    # the automatic verdict, but any individual frame that looked bad should still reach the vision
    # review pass. Loosening the aggregate must not also blind the second opinion.
    agg = _aggregate(per_frame, config.VIDEO_AGG_PCT)
    return agg, suspicious, len(fr), complete


def _video_keys(row):
    """The column-derived keys — source and posters. These go to the quarantine prefix, which the
    admin dashboard signs a read of so a human can judge an appeal.

    The pre-versioning preview (videos/previews/<id>-preview.mp4) is named here too, because it sits
    outside the versioned prefix _video_tree_keys lists — but DERIVED from the id, not read from
    previewUrl, which is null for long stretches while the file it named lives on.
    """
    keys = [r2.url_to_key(row.get(k)) for k in ("videoUrl", "thumbnailUrl", "thumbnailSmallUrl")]
    # ⚠️ previewUrl and hlsUrl are deliberately NOT read from their columns here.
    #
    # quarantine_keys runs FIRST, and that prefix is on the PUBLIC bucket. Taking the current preview
    # and master.m3u8 from their columns would copy them there — and move_to_private, running after,
    # would no longer find them. The result is the same object under two policies decided by which
    # call ran first: the superseded versions end up private while the CURRENT one stays publicly
    # readable at a derivable address, which is the worse half.
    #
    # The versioned shapes are swept by prefix in _video_tree_keys. Only the pre-versioning FLAT
    # preview needs naming, because it sits outside every prefix — derived, not read from the column,
    # for the same reason the ladder is: the column is null for long stretches. Mirrors vre-life.
    vid = row.get("id")
    if vid:
        keys.append(f"videos/previews/{vid}-preview.mp4")
    return keys


def _take_down_video(row, thumb_keys):
    """Move every public byte of a refused video out of reach.

    A function rather than inline, so a test can drive it: the alternative is grepping main.py for the
    call, which sees POSITION and not control flow — dedent the sweep out of the `rejected` branch and
    a text check still passes while approved videos get taken down.

    Two destinations, deliberately. The column-derived keys go to the quarantine prefix, which the
    admin dashboard signs a read of so a human can judge an appeal. The streaming tree goes to the
    PRIVATE bucket, because that prefix lives on the PUBLIC bucket and a set of streaming pieces copied
    there is a working video at a derivable address, not an obscure link.
    """
    r2.quarantine_keys(_video_keys(row) + thumb_keys)

    # ⚠️ The count is checked. Unlike quarantine_keys — which force-deletes the public copy when a copy
    # fails, so nothing is ever left readable — this mover leaves the object alone and returns a
    # number. The verdict is already terminal by now and nothing retries this row, so a quiet shortfall
    # is a permanent public leak.
    tree = _video_tree_keys(row)
    moved = r2.move_to_private(tree)
    if moved != len(tree):
        _log(f"[video] 🔴 TAKEDOWN INCOMPLETE id={row.get('id')}: {moved}/{len(tree)} moved — "
             f"the rest are STILL PUBLIC and nothing will retry this row")
    return moved


def _video_tree_keys(row):
    """The HLS ladder and the versioned previews, listed from the bucket and keyed on the VIDEO ID.

    ▶ Keyed on the id, never on hlsUrl. This used to be hlsUrl.rsplit("/", 1)[0] — the directory
      holding master.m3u8. Two ways that breaks: hlsUrl is NULL for long stretches, and since
      vre-video-worker began writing each transcode to videos/hls/<id>/<version>/ that expression
      names the CURRENT version only, leaving every superseded one publicly readable for ever. The id
      covers every version and every layout, including the pre-versioning one.

    ▶ Separate from _video_keys because these go somewhere ELSE — the private bucket, not the public
      quarantine prefix. See r2.move_to_private.

    ⚠️ UNREACHABLE TODAY, and that is why this is defence in depth rather than a fix. claim_video
      takes rows at moderationStatus='pending', which is BEFORE the transcode worker will touch them
      (it requires 'approved'), and nothing anywhere resets a ready video to pending. So when this
      loop sees a row, both prefixes are empty and hlsUrl is null. It must be correct before anything
      makes that path live, not after.
    """
    vid = row.get("id")
    if not vid:
        return []
    return r2.list_prefix(f"videos/hls/{vid}/") + r2.list_prefix(f"videos/previews/{vid}/")


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
        agg, suspicious, n, complete = res
        d = gate.decide_video(agg, suspicious)                    # clean → free; elegance band → vision
        t_status, t_scores, _media = _scan_thumbnail(row.get("postId"))   # CR2
        status = _worst(d["status"], t_status)
        # Truncated sampling must never produce a clean pass: we did not see the whole clip, so
        # "nothing found" is not evidence of nothing being there. Fail toward human review rather
        # than either approving unseen footage or rejecting a video on partial evidence.
        if not complete:
            status = _worst(status, "flagged")
        labels = dict(agg); labels["_reason"] = d.get("reason", ""); labels["_layer"] = d.get("layer", "")
        if not complete:
            labels["_incomplete_coverage"] = True
        labels.update({"thumb_" + k: v for k, v in t_scores.items()})
        db.resolve_video(row["id"], row.get("postId"), status, round(d.get("score", 0.0), 4), labels)
    except Exception as e:
        _log(f"[video] scan failed id={row['id']}: {type(e).__name__}: {e}")
        db.fail("Video", row["id"], row["moderationAttempts"])
        return
    if status == "rejected":
        thumb_keys = [r2.url_to_key(u) for u in (db.post_media_all_urls(row.get("postId")) or [])]
        _take_down_video(row, thumb_keys)
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
        agg, suspicious, n, complete = res
        d = gate.decide_video(agg, suspicious)   # SFW → 'approved' publishes the raw MP4 (no transcode)
        t_status, t_scores, _media = _scan_thumbnail(row["id"])   # CR2
        status = _worst(d["status"], t_status)
        # Same rule as the Video path, and it matters MORE here: an approved orphan post publishes
        # the raw MP4 straight away, with no transcode step in between.
        if not complete:
            status = _worst(status, "flagged")
        labels = dict(agg); labels["_reason"] = d.get("reason", ""); labels["_layer"] = d.get("layer", "")
        if not complete:
            labels["_incomplete_coverage"] = True
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
def process_media_scan(row):
    """One cover image awaiting a verdict, uploaded by someone sitting in a modal right now.

    Reads from the PRIVATE bucket by key — these bytes have no public URL, which is the whole point:
    a rejected cover never touched public storage, so there is nothing to take down afterwards.
    """
    t0 = time.time()
    try:
        data = r2.fetch_private(row["key"], max_bytes=config.MAX_IMAGE_BYTES)
        im = Image.open(io.BytesIO(data)).convert("RGB")
        d = gate.decide_image(im, api_key=config.ANTHROPIC_API_KEY, free_block=True)
    except Exception as e:
        _log(f"[scan] failed id={row['id']}: {type(e).__name__}: {e}")
        db.fail_media_scan(row["id"], row["attempts"])
        return

    status = d["status"]
    # `flagged` needs splitting, and getting this wrong takes the whole feature down.
    #
    # gate.decide_image degrades FAIL-CLOSED: when the vision call errors it returns 'flagged' with
    # layer='fallback'. So during an Anthropic outage EVERY storefront upload would come back
    # flagged, and a naive pass-through would tell every user their image "was not approved" while
    # silently blocking all service and bounty creation platform-wide.
    if d.get("layer") == "fallback":
        status = "error"          # our problem, not theirs — retryable, and the modal says so
    elif status == "flagged":
        # There is no human review queue for this surface (unlike Post/Video), so a hold nobody can
        # release is a dead end. Treat it as not-publishable and let them pick another image.
        status = "rejected"

    labels = dict(d.get("scores") or {})
    labels["_reason"] = d.get("reason", "")
    labels["_layer"] = d.get("layer", "")
    db.resolve_media_scan(row["id"], status, round(d.get("score", 0.0) or 0.0, 4), labels)

    # Nothing was ever public, so a rejection is a plain delete rather than a quarantine.
    if status != "approved":
        r2.delete_private([row["key"], row.get("keySmall")])

    _log(f"[scan] {row['id']} -> {status.upper()} [{d.get('layer')}:{d.get('reason','')}] "
         f"{time.time()-t0:.1f}s")


def media_scan_loop():
    """Own thread, on the IMAGE cadence. Deliberately not folded into image_loop: a 17-minute video
    scan must never sit in front of someone waiting in a modal."""
    while not _stop.is_set():
        try:
            row = db.claim_media_scan()
            if row:
                process_media_scan(row)
                continue
        except Exception:
            traceback.print_exc()
        _stop.wait(config.IMAGE_POLL_SEC)


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
    # Both bands are printed. There are now two, and a silently-wrong one is exactly how a chart
    # video ended up rejected for nudity — if the numbers in the log look wrong, they ARE wrong.
    print(f"[boot] nsfw-worker (NudeNet v3); IMAGE block={config.BLOCK_THRESHOLDS}", flush=True)
    print(f"[boot] VIDEO block={config.VIDEO_BLOCK_THRESHOLDS}", flush=True)
    print(f"[boot] VIDEO flag={config.VIDEO_FLAG_THRESHOLDS} agg_pct={config.VIDEO_AGG_PCT}", flush=True)
    if not config.DIRECT_URL:
        print("[boot] FATAL: DIRECT_URL unset", flush=True)
        sys.exit(1)
    nsfw.warmup()
    threading.Thread(target=_serve_metrics, daemon=True).start()
    threading.Thread(target=image_loop, daemon=True).start()
    threading.Thread(target=video_loop, daemon=True).start()
    threading.Thread(target=media_scan_loop, daemon=True).start()
    print(f"[boot] loops running; /metrics on :{config.METRICS_PORT}", flush=True)
    try:
        while not _stop.is_set():
            time.sleep(3600)
    except KeyboardInterrupt:
        _stop.set()


if __name__ == "__main__":
    main()
