#!/usr/bin/env python3
"""What a rejected video's bytes are, and where they go.

▶ WHY THIS FILE EXISTS. This repo had no test of its own, and a mutation pass proved the cost: every
  single breakage of the key-selection logic survived, because nothing anywhere executed it. That
  logic decides what stays publicly readable after a human or this worker refuses a video.

▶ NO PYTEST ON PURPOSE. This box is hand-deployed over ssh and its requirements are installed by hand;
  adding a test-only dependency to requirements.txt would mean a deploy to run a test. Plain asserts
  and a non-zero exit are enough, and match test_local.py, which `npm test` already runs.

    python3 test_keys.py
"""
import sys
import types

# Stub every module main.py imports, so this runs with no network, no credentials, no model weights
# and no Pillow. Only the two key-selection functions are under test; everything else is scenery.
# boto3 is installed on the box, not necessarily here. Stubbed so the REAL r2.py can be imported and
# its byte-moving logic driven — the alternative is skipping the only test of the code that decides
# whether a refused video's bytes actually leave public storage.
for _mod in ("boto3", "botocore", "botocore.config", "botocore.exceptions", "requests"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
sys.modules["boto3"].client = lambda *a, **k: None
sys.modules["botocore.config"].Config = lambda *a, **k: None
sys.modules["botocore.exceptions"].ClientError = type("ClientError", (Exception,), {})
sys.modules["botocore"].config = sys.modules["botocore.config"]
sys.modules["botocore"].exceptions = sys.modules["botocore.exceptions"]

PIL = types.ModuleType("PIL")
PIL.Image = types.SimpleNamespace(open=lambda *a, **k: None)
sys.modules.setdefault("PIL", PIL)
sys.modules.setdefault("PIL.Image", PIL.Image)

for name, attrs in {
    "r2": {"url_to_key": lambda u: None if not u else str(u).replace("https://media.vre.pro/", ""),
           "list_prefix": lambda p: [], "quarantine_keys": lambda k: None,
           "move_to_private": lambda k: None, "fetch_public": None, "download_to": None,
           "un_quarantine_keys": None, "fetch_private": None, "delete_private": None},
    "db": {}, "config": {"MAX_IMAGE_BYTES": 1}, "frames": {}, "gate": {}, "nsfw": {},
    "vision": {}, "adam": {}, "sysmetrics": {},
}.items():
    if name not in sys.modules:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

import r2 as r2stub                      # noqa: E402  the stub installed above
import main                              # noqa: E402

failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label} {detail}")
        failures.append(label)


VID = "a1b2c3d4-0000-4000-8000-000000000001"
listed = []
r2stub.list_prefix = lambda p: (listed.append(p) or [p + "x.ts"])

print("_video_tree_keys: the streaming tree")

# Keyed on the ID, never on hlsUrl. hlsUrl is NULL for long stretches, and since the transcoder began
# writing each encode to videos/hls/<id>/<version>/ it names only the CURRENT version — every
# superseded one would stay publicly readable for ever.
listed.clear()
main._video_tree_keys({"id": VID, "hlsUrl": None})
check("a NULL hlsUrl still sweeps both trees", listed == [f"videos/hls/{VID}/", f"videos/previews/{VID}/"], listed)

listed.clear()
main._video_tree_keys({"id": VID, "hlsUrl": f"https://media.vre.pro/videos/hls/{VID}/vabc/master.m3u8"})
check("a versioned hlsUrl does not narrow the sweep to that version",
      listed == [f"videos/hls/{VID}/", f"videos/previews/{VID}/"], listed)

listed.clear()
check("no id means no sweep, not a sweep of everything", main._video_tree_keys({"id": None}) == [] and listed == [])

print("_video_keys: the column-derived keys")
keys = main._video_keys({
    "id": VID,
    "videoUrl": "https://media.vre.pro/videos/a.mp4",
    "previewUrl": "https://media.vre.pro/videos/previews/" + VID + "-preview.mp4",
    "thumbnailUrl": None, "thumbnailSmallUrl": None, "hlsUrl": None,
})
check("the pre-versioning flat preview is still reached via its column",
      f"videos/previews/{VID}-preview.mp4" in keys, keys)
# It must NOT list — that is the other function's job, and doing both would send the same object to
# two destinations depending on which ran first.
listed.clear()
main._video_keys({"id": VID, "videoUrl": None, "previewUrl": None, "thumbnailUrl": None, "thumbnailSmallUrl": None, "hlsUrl": None})
check("the column path does not list the bucket", listed == [], listed)

print("_take_down_video: where the bytes actually go")

# A recording stand-in for the two movers, so this drives the real branch instead of grepping for it.
# Grepping sees position, not control flow: dedent the sweep out of the rejected branch and a text
# check still passes while approved videos get taken down.
quarantined, privated = [], []
r2stub.quarantine_keys = lambda k: quarantined.extend(k)
r2stub.move_to_private = lambda k: (privated.extend(k) or len(k))
r2stub.list_prefix = lambda p: [p + "seg.ts"]

ROW = {
    "id": VID,
    "videoUrl": "https://media.vre.pro/videos/src.mp4",
    "previewUrl": f"https://media.vre.pro/videos/previews/{VID}/vabc/preview.mp4",
    "thumbnailUrl": "https://media.vre.pro/videos/thumb.jpg",
    "thumbnailSmallUrl": None,
    "hlsUrl": f"https://media.vre.pro/videos/hls/{VID}/vabc/master.m3u8",
}
main._take_down_video(ROW, ["videos/postthumb.jpg"])
# url_to_key returns None for an absent column; quarantine_keys skips falsy keys, so filter here too.
quarantined = [k for k in quarantined if k]

check("the source MP4 is taken", "videos/src.mp4" in quarantined, quarantined)
check("the poster is taken", "videos/thumb.jpg" in quarantined, quarantined)
check("post thumbnails are taken", "videos/postthumb.jpg" in quarantined, quarantined)
# The one that matters: the CURRENT versioned preview and master must not reach the public prefix.
# quarantine_keys runs first, so taking them from their columns would copy them there and the private
# mover would no longer find them — the superseded versions end up private while the current one stays
# publicly readable, which is the worse half.
check("the current versioned preview does NOT go to the public prefix",
      not any(f"videos/previews/{VID}/" in k for k in quarantined), quarantined)
check("the current master does NOT go to the public prefix",
      not any(f"videos/hls/{VID}/" in k for k in quarantined), quarantined)
check("the pre-versioning flat preview IS named (no prefix reaches it)",
      f"videos/previews/{VID}-preview.mp4" in quarantined, quarantined)
check("both streaming trees go to the private bucket",
      privated == [f"videos/hls/{VID}/seg.ts", f"videos/previews/{VID}/seg.ts"], privated)

print("move_to_private: the real mover")
import r2 as _r2mod
sys.modules.pop("r2", None)
import importlib.util
spec = importlib.util.spec_from_file_location("realr2", "r2.py")
# config is stubbed above; give the real module what it reads.
sys.modules["config"].R2_BUCKET_NAME = "pub"
sys.modules["config"].R2_PRIVATE_BUCKET = "priv"
sys.modules["config"].R2_ENDPOINT = "http://127.0.0.1:1"
sys.modules["config"].R2_ACCESS_KEY_ID = "x"
sys.modules["config"].R2_SECRET_ACCESS_KEY = "y"
try:
    realr2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(realr2)

    ops = []

    class FakeS3:
        def copy_object(self, **kw):
            ops.append(("copy", kw["Bucket"], kw["Key"], kw["CopySource"]["Bucket"]))

        def delete_object(self, **kw):
            ops.append(("delete", kw["Bucket"], kw["Key"]))

    realr2._client = lambda: FakeS3()
    moved = realr2.move_to_private(["a.ts", "b.ts"])
    check("every key is moved", moved == 2, moved)
    check("copy targets the PRIVATE bucket", all(o[1] == "priv" for o in ops if o[0] == "copy"), ops)
    check("copy reads from the PUBLIC bucket", all(o[3] == "pub" for o in ops if o[0] == "copy"), ops)
    check("delete targets the PUBLIC bucket", all(o[1] == "pub" for o in ops if o[0] == "delete"), ops)
    # Copy BEFORE delete, per key. Reversed, the only copy is destroyed.
    for k in ("a.ts", "b.ts"):
        ci = next(i for i, o in enumerate(ops) if o[0] == "copy" and o[2] == k)
        di = next(i for i, o in enumerate(ops) if o[0] == "delete" and o[2] == k)
        check(f"{k} is copied before it is deleted", ci < di, ops)

    class Boom(Exception):
        def __init__(self, code):
            self.response = {"Error": {"Code": code}}

    class FailS3(FakeS3):
        code = "NoSuchKey"

        def copy_object(self, **kw):
            raise Boom(self.code)

    f = FailS3()
    realr2._client = lambda: f
    check("a key already gone counts as moved", realr2.move_to_private(["a.ts"]) == 1)
    f.code = "NoSuchBucket"
    check("a wrong BUCKET does NOT count as moved", realr2.move_to_private(["a.ts"]) == 0,
          "a misconfigured destination would report a clean takedown of bytes still public")
except Exception as e:                                  # noqa: BLE001
    check("the real r2 module loads", False, f"{type(e).__name__}: {e}")

print("control flow: the takedown runs ONLY for a refused video")

# Parsed, not grepped. A text search sees POSITION: dedent the call out of the rejected branch and it
# still appears in the file, still after the verdict, and a grep-based check still passes — while
# every APPROVED video quietly gets its bytes taken down.
import ast

tree = ast.parse(open("main.py").read())
fn = next(n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "process_video")


def calls_in(node):
    return [d.func.id for d in ast.walk(node)
            if isinstance(d, ast.Call) and isinstance(d.func, ast.Name)]


guarded = [n for n in ast.walk(fn)
           if isinstance(n, ast.If)
           and isinstance(n.test, ast.Compare)
           and isinstance(n.test.left, ast.Name) and n.test.left.id == "status"
           and any(isinstance(c, ast.Constant) and c.value == "rejected" for c in n.test.comparators)]
check("process_video has a rejected branch", len(guarded) == 1, len(guarded))
inside = any("_take_down_video" in calls_in(b) for g in guarded for b in g.body)
check("the takedown is INSIDE the rejected branch", inside)

# …and nowhere else in the function, which is what a dedent would produce.
everywhere = calls_in(fn).count("_take_down_video")
check("the takedown is called exactly once, and only there", everywhere == 1 and inside, everywhere)

print()
if failures:
    print(f"FAILED: {len(failures)}")
    sys.exit(1)
print("all checks passed")
