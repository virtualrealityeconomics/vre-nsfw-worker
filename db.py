"""Postgres poller for the NSFW worker.

Claims are lease-based (`moderationLockedAt`) + `FOR UPDATE SKIP LOCKED`, so MANY identical worker
instances can run at once (scale by deploying more replicas) and a crashed instance self-heals: its
lease goes stale and the row is re-claimable. Autocommit — each claim/verdict is one statement whose
lock releases instantly.

Agreed flow (differs from the original audit, deliberately):
  IMAGE post  → created 'approved' (optimistic, already public); we scan un-scanned ones
                (moderationStatus='approved' AND moderatedAt IS NULL). NSFW → flip to 'rejected'/'flagged'
                (hides it) + quarantine. SFW → stamp moderatedAt (stays 'approved').
  VIDEO       → created 'pending' (hidden). SFW → Video.moderationStatus='approved' (the Vultr transcode
                worker will claim ONLY approved videos, then flip the Post public). NSFW → Video+Post
                'rejected' + quarantine.
  ORPHAN video post (video enroll failed → no Video row) → scan Post.videoUrl directly; SFW → Post
                'approved' (plays raw MP4, no HLS); NSFW → 'rejected'.
"""
import threading

import psycopg2
import psycopg2.extras

import config

_local = threading.local()


def _conn():
    c = getattr(_local, "conn", None)
    if c is not None:
        try:
            if c.closed == 0:
                return c
        except Exception:
            pass
    c = psycopg2.connect(config.DIRECT_URL)
    c.autocommit = True
    _local.conn = c
    return c


def _exec(sql, params=None, fetch=None):
    for attempt in (1, 2):
        try:
            with _conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params or ())
                if fetch == "one":
                    return cur.fetchone()
                if fetch == "all":
                    return cur.fetchall()
                return None
        except psycopg2.Error:
            try:
                _local.conn.close()
            except Exception:
                pass
            _local.conn = None
            if attempt == 2:
                raise


# Lease predicate: unlocked OR the lease expired (crashed worker) → re-claimable.
_LEASE = '("moderationLockedAt" IS NULL OR "moderationLockedAt" < NOW() - (%s * INTERVAL \'1 minute\'))'


# ── Claims ──────────────────────────────────────────────────────────────────────────────────────
def claim_image_post():
    sql = f"""
        UPDATE "Post" SET "moderationLockedAt" = NOW()
        WHERE id = (
            SELECT id FROM "Post"
            WHERE "moderationStatus" = 'approved' AND "moderatedAt" IS NULL
              AND array_length("mediaUrls", 1) >= 1 AND ("videoUrl" IS NULL OR "videoUrl" = '')
              AND "moderationAttempts" < %s AND {_LEASE}
            ORDER BY "createdAt" ASC LIMIT 1 FOR UPDATE SKIP LOCKED
        )
        RETURNING id, "userId", "mediaUrls", "mediaUrlsSmall", "mediaUrlsMedium", "moderationAttempts"
    """
    return _exec(sql, (config.MAX_ATTEMPTS, config.LEASE_MINUTES), fetch="one")


def claim_video():
    sql = f"""
        UPDATE "Video" SET "moderationLockedAt" = NOW()
        WHERE id = (
            SELECT id FROM "Video"
            WHERE "moderationStatus" = 'pending' AND "moderationAttempts" < %s AND {_LEASE}
            ORDER BY "createdAt" ASC LIMIT 1 FOR UPDATE SKIP LOCKED
        )
        RETURNING id, "postId", "videoUrl", "previewUrl", "thumbnailUrl", "thumbnailSmallUrl", "hlsUrl", "moderationAttempts"
    """
    return _exec(sql, (config.MAX_ATTEMPTS, config.LEASE_MINUTES), fetch="one")


def claim_orphan_video_post():
    """A post with a videoUrl but NO Video row (best-effort enroll failed) — both other loops miss it."""
    sql = f"""
        UPDATE "Post" SET "moderationLockedAt" = NOW()
        WHERE id = (
            SELECT p.id FROM "Post" p
            WHERE p."moderationStatus" = 'pending' AND p."videoUrl" IS NOT NULL AND p."videoUrl" <> ''
              AND p."moderationAttempts" < %s AND {_LEASE.replace('"moderationLockedAt"', 'p."moderationLockedAt"')}
              AND NOT EXISTS (SELECT 1 FROM "Video" v WHERE v."postId" = p.id)
            ORDER BY p."createdAt" ASC LIMIT 1 FOR UPDATE SKIP LOCKED
        )
        RETURNING id, "userId", "videoUrl", "moderationAttempts"
    """
    return _exec(sql, (config.MAX_ATTEMPTS, config.LEASE_MINUTES), fetch="one")


def post_media_urls(post_id):
    """A post's client-supplied images (CR2: a video post's thumbnail publishes as the poster but no
    image loop scans it — the video loop scans it separately). Full-size only → what we SCAN."""
    if not post_id:
        return []
    r = _exec('SELECT "mediaUrls" FROM "Post" WHERE id=%s', (post_id,), fetch="one")
    return (r or {}).get("mediaUrls") or []


def post_media_all_urls(post_id):
    """R2: the FULL derivative set (mediaUrls + Small + Medium) for a video post's thumbnail — what we
    QUARANTINE on reject, so the 320/800px NSFW thumbnails don't stay public while the full-size is
    pulled. Scanning still uses only full-size (post_media_urls)."""
    if not post_id:
        return []
    r = _exec(
        'SELECT "mediaUrls", "mediaUrlsSmall", "mediaUrlsMedium" FROM "Post" WHERE id=%s',
        (post_id,), fetch="one",
    ) or {}
    seen, out = set(), []
    for col in ("mediaUrls", "mediaUrlsSmall", "mediaUrlsMedium"):
        for u in (r.get(col) or []):
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    return out


def describe(post_id):
    """Author + a short title for the scan log ("what am I testing"). ONE cheap SELECT, FULLY best-effort
    — a cosmetic log helper must NEVER raise into the scan path. Columns match schema.prisma
    (Post.content/userId, User.username/handle)."""
    default = {"author": "?", "title": ""}
    if not post_id:
        return default
    try:
        r = _exec(
            'SELECT COALESCE(u.username, u.handle, \'?\') AS author, LEFT(COALESCE(p.content, \'\'), 40) AS title '
            'FROM "Post" p LEFT JOIN "User" u ON u.id = p."userId" WHERE p.id=%s',
            (post_id,), fetch="one",
        )
        return r or default
    except Exception:
        return default


# ── Verdicts ────────────────────────────────────────────────────────────────────────────────────
def _notify_hidden(post_id):
    """Tell the owner (once) their post is under review — fires on flag/reject. Best-effort; the
    backoffice badge is the primary signal, so a failed insert never blocks the verdict."""
    if not post_id:
        return
    try:
        _exec(
            'INSERT INTO "Notification" (id,"userId",type,title,"entityType","entityId","linkPath","actorType","createdAt") '
            "SELECT gen_random_uuid(), p.\"userId\", 'content_flagged', 'Your content is under review', "
            "'post', p.id, '/posts?backoffice=true&tab=my-posts&highlight='||p.id, 'system', NOW() "
            'FROM "Post" p WHERE p.id=%s '
            "AND NOT EXISTS (SELECT 1 FROM \"Notification\" n WHERE n.type='content_flagged' AND n.\"entityId\"=p.id)",
            (post_id,),
        )
    except Exception as e:
        print(f"[notify] content_flagged failed post={post_id}: {type(e).__name__}: {e}", flush=True)


def resolve_post(post_id, status, score, labels):
    _exec(
        'UPDATE "Post" SET "moderationStatus"=%s, "moderatedAt"=NOW(), "moderationScore"=%s, '
        '"moderationLabels"=%s, "moderationLockedAt"=NULL WHERE id=%s',
        (status, score, psycopg2.extras.Json(labels) if labels is not None else None, post_id),
    )
    if status in ("flagged", "rejected"):
        _notify_hidden(post_id)


def resolve_video(video_id, post_id, status, score, labels):
    # On reject, hide the linked Post FIRST (so it's never public), then stamp the Video. Crash-safe:
    # if it dies between, the Video stays 'pending' → re-scanned → same verdict (idempotent).
    if status == "rejected" and post_id:
        _exec(
            'UPDATE "Post" SET "moderationStatus"=%s, "moderatedAt"=NOW(), "moderationScore"=%s WHERE id=%s',
            ("rejected", score, post_id),
        )
    _exec(
        'UPDATE "Video" SET "moderationStatus"=%s, "moderatedAt"=NOW(), "moderationScore"=%s, '
        '"moderationLabels"=%s, "moderationLockedAt"=NULL WHERE id=%s',
        (status, score, psycopg2.extras.Json(labels) if labels is not None else None, video_id),
    )
    if status in ("flagged", "rejected"):
        _notify_hidden(post_id)


def fail(table, row_id, current_attempts):
    """Scan error → bump attempts + release the lease. At the cap, flip to terminal 'error' (hidden,
    surfaced to admin/uploader), never invisibly stuck."""
    _exec(
        f'UPDATE "{table}" SET "moderationAttempts"="moderationAttempts"+1, '
        f'"moderationStatus"=CASE WHEN "moderationAttempts"+1 >= %s THEN \'error\' ELSE "moderationStatus" END, '
        f'"moderatedAt"=CASE WHEN "moderationAttempts"+1 >= %s THEN NOW() ELSE "moderatedAt" END, '
        f'"moderationLockedAt"=NULL WHERE id=%s',
        (config.MAX_ATTEMPTS, config.MAX_ATTEMPTS, row_id),
    )


# ── Metrics (health: a backed-up/down worker shows as a growing backlog) ──────────────────────────
def metrics():
    out = {}
    rows = _exec('SELECT "moderationStatus" AS s, COUNT(*) AS n FROM "Post" GROUP BY 1', fetch="all") or []
    out["post"] = {r["s"]: r["n"] for r in rows}
    rows = _exec('SELECT "moderationStatus" AS s, COUNT(*) AS n FROM "Video" GROUP BY 1', fetch="all") or []
    out["video"] = {r["s"]: r["n"] for r in rows}
    # Actionable backlog = the two things the worker owes: unscanned optimistic images + pending videos.
    r = _exec(
        'SELECT (SELECT COUNT(*) FROM "Post" WHERE "moderationStatus"=\'approved\' AND "moderatedAt" IS NULL '
        'AND array_length("mediaUrls",1)>=1 AND ("videoUrl" IS NULL OR "videoUrl"=\'\')) AS image_backlog, '
        '(SELECT COUNT(*) FROM "Video" WHERE "moderationStatus"=\'pending\') AS video_backlog',
        fetch="one",
    ) or {}
    out["backlog"] = {"images": r.get("image_backlog", 0), "videos": r.get("video_backlog", 0)}
    return out
