"""R2 access: fetch media from the PUBLIC url (no creds), and QUARANTINE blocked bytes.

Quarantine, never hard-delete (audit C3 + never-hard-delete rule): copy the object to a private
`quarantine/` prefix, THEN delete the public-domain copy — so the URL 404s but the bytes survive
for appeal/restore. If the copy fails we still delete the public copy (an NSFW object must not stay
public), logging loudly.
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


def quarantine_keys(keys):
    """Copy each key to the quarantine prefix, then delete the public copy. Missing keys skipped."""
    c = _client()
    moved = 0
    for key in keys:
        if not key:
            continue
        dest = config.QUARANTINE_PREFIX + key
        try:
            c.copy_object(
                Bucket=config.R2_BUCKET_NAME,
                CopySource={"Bucket": config.R2_BUCKET_NAME, "Key": key},
                Key=dest,
            )
            c.delete_object(Bucket=config.R2_BUCKET_NAME, Key=key)
            moved += 1
        except Exception as e:
            # Do NOT leave the public copy live — an NSFW object must 404 even if the appeal-copy failed.
            print(f"[r2] quarantine {key} failed ({type(e).__name__}: {e}); force-deleting public copy", flush=True)
            try:
                c.delete_object(Bucket=config.R2_BUCKET_NAME, Key=key)
            except Exception:
                pass
    return moved


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
