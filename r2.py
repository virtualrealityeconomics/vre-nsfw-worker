"""R2 access: fetch media from the PUBLIC url (no creds), and QUARANTINE blocked bytes.

Quarantine, never hard-delete (audit C3 + never-hard-delete rule): copy the object to the PRIVATE
bucket under a `quarantine/` prefix, THEN delete the public-domain copy — so the URL 404s but the
bytes survive for appeal/restore.

⚠️ The quarantine used to live under that prefix on the PUBLIC bucket, which changed an object's
address without changing who could read it: 137 objects sat there readable by anyone who could derive
the URL, held back only by a Cloudflare rule. The prefix is unchanged; only the bucket moved.
"""
import boto3
import requests

import config

_s3 = None


def _client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client(
            "s3",
            endpoint_url=config.R2_ENDPOINT,
            aws_access_key_id=config.R2_ACCESS_KEY_ID,
            aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
            region_name="auto",
        )
    return _s3


def url_to_key(url):
    """`https://media.vre.pro/<key>` → `<key>`. Tolerates a bare key or a different host."""
    if not url:
        return None
    if url.startswith(config.MEDIA_BASE):
        return url[len(config.MEDIA_BASE):]
    if "://" in url:
        parts = url.split("/", 3)
        return parts[3] if len(parts) >= 4 else None
    return url.lstrip("/")


def fetch_public(url, max_bytes=None):
    """Download bytes from the public URL. Streams with a size cap (decode-DoS guard)."""
    r = requests.get(url, timeout=45, stream=True)
    r.raise_for_status()
    data = bytearray()
    for chunk in r.iter_content(65536):
        data.extend(chunk)
        if max_bytes and len(data) > max_bytes:
            raise ValueError("media exceeds max bytes")
    return bytes(data)


def download_to(url, path, max_bytes=None):
    """Stream a (large) file to disk — used for video before ffmpeg."""
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    total = 0
    with open(path, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            total += len(chunk)
            if max_bytes and total > max_bytes:
                raise ValueError("media exceeds max bytes")
            f.write(chunk)
    return total


def _private_object_exists(c, key):
    """Is the object actually in the private bucket? Used to decide whether a public copy is redundant.

    boto3 raises ClientError with Code "404" for a missing KEY on head_object — NOT "NoSuchKey" and NOT
    "NotFound". A literal port of the JS predicate would never return True, so the gate would always say
    "not there" and nothing would ever be cleaned up.

    Catching Exception, not ClientError: botocore's connection failures (EndpointConnectionError and
    friends) are BotoCoreError, NOT ClientError, so a narrow catch would let a network blip escape a
    function whose whole job is to answer yes or no. Any error at all -> False -> do not delete. A
    missing bucket, a permissions problem and a genuinely absent object all mean the same thing here:
    we have no proof the bytes are safe anywhere else.
    """
    try:
        c.head_object(Bucket=config.R2_PRIVATE_BUCKET, Key=key)
        return True
    except Exception:                                   # noqa: BLE001 — see below
        return False


def quarantine_keys(keys):
    """Copy each key into the PRIVATE bucket's quarantine prefix, then delete the public copy.

    ⛔ THE DELETE IS EARNED, NOT ASSUMED. This used to force-delete the public copy on ANY error, which
    was safe while the copy was within one bucket — the only thing that can be missing then is the
    source key. Cross-bucket, the same error class also means the destination is missing or unwritable,
    and the old test could not tell them apart. Measured against real R2:

        a valid-but-absent bucket -> NoSuchBucket, "The specified bucket does not exist."
            ...the old substring test matched -> `continue` -> the NSFW bytes stayed PUBLIC
        an invalid bucket name    -> InvalidBucketName
            ...it did not match   -> the force-delete fired -> THE ONLY COPY WAS DESTROYED

    A coin flip between leaving blocked content public and destroying a user's media. So: match on the
    error CODE, never the message string, and delete only once the bytes are provably somewhere else.
    Leaving an NSFW object public for a few more minutes is recoverable; deleting the only copy is not,
    and this project never hard-deletes.
    """
    c = _client()
    moved = already_gone = failed = 0
    errors = []
    for key in keys:
        if not key:
            continue
        dest = config.QUARANTINE_PREFIX + key
        try:
            c.copy_object(
                Bucket=config.R2_PRIVATE_BUCKET,
                CopySource={"Bucket": config.R2_BUCKET_NAME, "Key": key},
                Key=dest,
            )
            c.delete_object(Bucket=config.R2_BUCKET_NAME, Key=key)
            moved += 1
        except Exception as e:                          # noqa: BLE001
            # Exception, not ClientError: botocore raises EndpointConnectionError and similar for
            # network failures, and those are NOT ClientError. A narrow catch would crash the worker
            # loop on a blip — the old code caught broadly and that part was right.
            code = getattr(e, "response", {}).get("Error", {}).get("Code") if hasattr(e, "response") else None
            # On the CODE. "The specified bucket does not exist" contains "does not exist" too, which is
            # exactly how a whole-configuration failure used to read as "this one object was already done".
            if code == "NoSuchKey":
                already_gone += 1
                continue
            if _private_object_exists(c, dest):
                # The copy DID land; the public copy is redundant. Only count it moved if the delete
                # actually succeeds — otherwise the object is still at its public URL and saying
                # "moved" is the lie this rewrite exists to remove.
                try:
                    c.delete_object(Bucket=config.R2_BUCKET_NAME, Key=key)
                    moved += 1
                except Exception as de:                 # noqa: BLE001
                    failed += 1
                    errors.append(f"{key}: copied but not deleted: {de}")
                    print(f"[r2] quarantine copied but PUBLIC COPY REMAINS {key}: {de}", flush=True)
                continue
            failed += 1
            errors.append(f"{key}: {e}")
            print(f"[r2] quarantine FAILED, object left PUBLIC: {key}: {e}", flush=True)
    return {"moved": moved, "already_gone": already_gone, "failed": failed, "errors": errors}


def un_quarantine_keys(keys):
    """Reverse of quarantine_keys (D7): copy each object back from the quarantine prefix to its public
    key, then delete the quarantine copy — used by the admin-approve action to restore a previously
    REJECTED item. Missing quarantine copies are skipped (idempotent)."""
    c = _client()
    restored = 0
    for key in keys:
        if not key:
            continue
        src = config.QUARANTINE_PREFIX + key
        try:
            c.copy_object(
                Bucket=config.R2_BUCKET_NAME,
                CopySource={"Bucket": config.R2_BUCKET_NAME, "Key": src},
                Key=key,
            )
            c.delete_object(Bucket=config.R2_BUCKET_NAME, Key=src)
            restored += 1
        except Exception as e:
            print(f"[r2] un-quarantine {key} failed ({type(e).__name__}: {e})", flush=True)
    return restored


def list_prefix(prefix):
    """All keys under a prefix (paginated) — for sweeping the variable HLS `.ts` segment set."""
    if not prefix:
        return []
    c = _client()
    keys = []
    token = None
    while True:
        kw = {"Bucket": config.R2_BUCKET_NAME, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = c.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            keys.append(o["Key"])
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys


# ── Private staging bucket ────────────────────────────────────────────────────────────────────────
# Cover images awaiting a verdict live here and have NO public URL, so they are read through the S3
# API by key rather than over HTTP. That is the point: a rejected cover never reached public storage,
# so unlike the quarantine path there is nothing to take down afterwards.
PRIVATE_BUCKET = config.R2_PRIVATE_BUCKET


def fetch_private(key, max_bytes=None):
    """Read an object out of the private bucket. Same size cap as fetch_public — a decompression
    bomb is just as dangerous whichever bucket it came from."""
    c = _client()
    obj = c.get_object(Bucket=PRIVATE_BUCKET, Key=key)
    body = obj["Body"]
    data = bytearray()
    while True:
        chunk = body.read(65536)
        if not chunk:
            break
        data.extend(chunk)
        if max_bytes and len(data) > max_bytes:
            raise ValueError("media exceeds max bytes")
    return bytes(data)


def move_to_private(keys):
    """Copy each key into the PRIVATE bucket, then delete the public copy. Missing keys skipped.

    Why not quarantine_keys: that prefix is on the PUBLIC bucket, so it changes an object's address
    without changing who can read it. For a lone MP4 that is an obscure link. For an HLS tree it is a
    complete, working stream — ffmpeg writes bare relative filenames into each variant playlist, so
    the whole set copied under quarantine/ plays perfectly at an address derivable from the original.
    The private bucket has no unsigned read path at all.

    vre-life's human reject already routes the ladder this way; this keeps the automatic path in step.
    """
    c = _client()
    moved = 0
    for key in keys:
        if not key:
            continue
        try:
            c.copy_object(
                Bucket=PRIVATE_BUCKET,
                CopySource={"Bucket": config.R2_BUCKET_NAME, "Key": key},
                Key=key,
            )
            # Copy first, delete second, per key — a crash can never lose the only copy.
            c.delete_object(Bucket=config.R2_BUCKET_NAME, Key=key)
            moved += 1
        except Exception as e:
            code = ""
            resp = getattr(e, "response", None)
            if isinstance(resp, dict):
                code = resp.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                moved += 1      # already gone from public = already moved; a re-run finishes a partial sweep
                continue
            print(f"[r2] move_to_private failed {key}: {type(e).__name__}: {e}", flush=True)
    return moved


def delete_private(keys):
    """Hard-delete from the private bucket — a rejected cover, or a swept orphan."""
    c = _client()
    n = 0
    for key in keys or []:
        if not key:
            continue
        try:
            c.delete_object(Bucket=PRIVATE_BUCKET, Key=key)
            n += 1
        except Exception as e:
            print(f"[r2] private delete failed {key}: {type(e).__name__}: {e}", flush=True)
    return n
