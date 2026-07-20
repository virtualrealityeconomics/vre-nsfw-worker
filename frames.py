"""Sample frames from a video for NSFW scanning (ffmpeg).

Strategy: a GUARANTEED evenly-spaced pass (>= MIN_FRAMES, <= MAX_FRAMES across the whole clip, based
on probed duration) so coverage never depends on the clip having scene cuts — plus a best-effort
scene-change pass (extra frames at cuts, where new content usually appears). Frames are downscaled
in ffmpeg (aspect preserved, long side 320 = NudeNet's input) and capped. Aggregation is MAX (in
main), so one explicit frame fails the whole video — sampling only needs to CATCH that frame.

Returns [] when ffmpeg yields nothing (corrupt/unreadable) → caller treats as an error/retry,
NEVER as a pass (fail-closed: we don't approve a video we couldn't actually look at).
"""
import glob
import os
import subprocess
import tempfile

from PIL import Image

import config

_SCALE = "scale=320:320:force_original_aspect_ratio=decrease"  # long side 320, aspect preserved


def _duration(video_path):
    """Seconds via ffprobe; 0.0 if it can't be determined."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", video_path],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _run_ffmpeg(video_path, vf, out_dir, prefix, cap, timeout=180):
    """Extract frames. stderr is swallowed on purpose — the scene pass legitimately produces nothing
    on cut-less clips and would otherwise spew encoder errors. A real failure just yields no files."""
    out = os.path.join(out_dir, f"{prefix}_%05d.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path,
        "-vf", vf, "-vsync", "vfr", "-frames:v", str(cap), out,
    ]
    try:
        subprocess.run(cmd, timeout=timeout, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        print(f"[frames] ffmpeg timeout on {prefix}", flush=True)


def sample(video_path):
    frames = []
    dur = _duration(video_path)
    with tempfile.TemporaryDirectory() as d:
        cap = config.MAX_FRAMES
        if dur and dur > 0:
            # Guarantee even coverage: target between MIN_FRAMES and MAX_FRAMES, spread across the clip.
            target = min(config.MAX_FRAMES, max(config.MIN_FRAMES, round(dur * config.FPS_FLOOR)))
            even_fps = max(target / dur, 0.01)
            _run_ffmpeg(video_path, f"fps={even_fps:.4f},{_SCALE}", d, "even", cap)
        else:
            # Unknown duration → fixed floor (still yields frames; just not count-guaranteed).
            _run_ffmpeg(video_path, f"fps={config.FPS_FLOOR},{_SCALE}", d, "even", cap)
        # Bonus: extra frames exactly at scene cuts (best-effort; may legitimately be empty).
        _run_ffmpeg(video_path, f"select='gt(scene,{config.SCENE_THRESHOLD})',{_SCALE}", d, "scene", cap)
        for fp in sorted(glob.glob(os.path.join(d, "*.jpg")))[: config.MAX_FRAMES]:
            try:
                im = Image.open(fp)
                im.load()
                frames.append(im.convert("RGB"))
            except Exception:
                continue
    return frames
