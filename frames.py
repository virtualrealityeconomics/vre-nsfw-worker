"""Sample frames from a video for NSFW scanning (ffmpeg).

Hybrid strategy (audit §3): scene-changes (where new content usually appears at a cut) + a uniform
time floor across the whole clip (so a static-then-explicit stretch can't slip between cuts). Frames
are downscaled to 224 in ffmpeg (cheap) and capped. Aggregation is MAX (in main), so one explicit
frame fails the whole video — sampling only needs to be dense enough to CATCH that frame.

Returns [] when ffmpeg yields nothing (corrupt/unreadable) → caller treats as an error/retry,
NEVER as a pass (fail-closed: we don't approve a video we couldn't actually look at).
"""
import glob
import os
import subprocess
import tempfile

from PIL import Image

import config


def _run_ffmpeg(video_path, vf, out_glob_dir, prefix, cap, timeout=180):
    out = os.path.join(out_glob_dir, f"{prefix}_%05d.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path,
        "-vf", vf, "-vsync", "vfr", "-frames:v", str(cap), out,
    ]
    try:
        subprocess.run(cmd, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        print(f"[frames] ffmpeg timeout on {prefix}", flush=True)


def sample(video_path):
    frames = []
    with tempfile.TemporaryDirectory() as d:
        cap = config.MAX_FRAMES
        # Scene changes (cuts) — where new/explicit content typically starts.
        _run_ffmpeg(video_path, f"select='gt(scene,{config.SCENE_THRESHOLD})',scale=224:224", d, "scene", cap)
        # Uniform floor across the whole video — guarantees coverage between cuts.
        _run_ffmpeg(video_path, f"fps={config.FPS_FLOOR},scale=224:224", d, "unif", cap)
        for fp in sorted(glob.glob(os.path.join(d, "*.jpg")))[: config.MAX_FRAMES]:
            try:
                im = Image.open(fp)
                im.load()
                frames.append(im.convert("RGB"))
            except Exception:
                continue
    return frames
