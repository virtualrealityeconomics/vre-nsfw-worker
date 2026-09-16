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
    "videoUrl": "https://media.vre.pro/videos/a.mp4",
    "previewUrl": "https://media.vre.pro/videos/previews/" + VID + "-preview.mp4",
    "thumbnailUrl": None, "thumbnailSmallUrl": None, "hlsUrl": None,
})
check("the pre-versioning flat preview is still reached via its column",
      f"videos/previews/{VID}-preview.mp4" in keys, keys)
# It must NOT list — that is the other function's job, and doing both would send the same object to
# two destinations depending on which ran first.
listed.clear()
main._video_keys({"videoUrl": None, "previewUrl": None, "thumbnailUrl": None, "thumbnailSmallUrl": None, "hlsUrl": None})
check("the column path does not list the bucket", listed == [], listed)

print("destination: the tree must NOT go to the public quarantine prefix")
src = open("main.py").read()
# Anchored on the VIDEO rejection specifically — there are three `if status == "rejected"` blocks in
# this file and the first is the image path, which has no streaming tree.
i = src.index("quarantine_keys(_video_keys(row)")
after = src[i:i + 400]
check("the tree is moved to the private bucket, in the video rejection",
      "move_to_private(_video_tree_keys(row))" in after, after[:120])
check("the tree is not handed to quarantine_keys",
      "quarantine_keys(_video_tree_keys" not in src,
      "quarantine/ is on the PUBLIC bucket — an HLS tree there is a working stream")

print()
if failures:
    print(f"FAILED: {len(failures)}")
    sys.exit(1)
print("all checks passed")
