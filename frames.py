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


def _run_ffmpeg(video_path, vf, out_dir, prefix, cap, timeout=180, keyframes_only=False):
    """Extract frames. Returns True if ffmpeg finished, False if it was killed on timeout.

    The caller MUST look at that flag. ffmpeg writes JPEGs progressively, so a timeout leaves a
    partial, START-BIASED set on disk that looks exactly like a successful small sample — the old
    code logged the timeout and then silently graded the video on whatever had been written. A
    1080p/17min clip yielded 115 of 150 frames, so the last ~4 minutes were never examined and the
    verdict was still reported as final.

    stderr is swallowed on purpose — the scene pass legitimately produces nothing on cut-less clips
    and would otherwise spew encoder errors.

    keyframes_only decodes just keyframes (`-skip_frame nokey`, before -i so it applies to the
    decoder). Measured on a 1054s 1080p file: 6.9s vs 228.6s for a full decode, covering 0→1040s
    versus 0→1047s — essentially the same coverage 33x faster. That speed IS the safety fix: the
    old path was slow enough to hit its own timeout.
    """
    out = os.path.join(out_dir, f"{prefix}_%05d.jpg")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if keyframes_only:
        cmd += ["-skip_frame", "nokey"]
    cmd += ["-i", video_path, "-vf", vf, "-vsync", "vfr", "-frames:v", str(cap), out]
    try:
        subprocess.run(cmd, timeout=timeout, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.TimeoutExpired:
        print(f"[frames] ffmpeg timeout on {prefix} after {timeout}s", flush=True)
        return False


def _timeout_for(dur):
    """Scale the budget with the clip. A backstop only — with keyframe decoding nothing should get
    close to it, but a pathological file must not be graded on a truncated sample."""
    if not dur or dur <= 0:
        return 300
    return int(max(180, min(1800, 60 + dur * 0.6)))


def _count(d, prefix):
    return len(glob.glob(os.path.join(d, f"{prefix}_*.jpg")))


def sample(video_path):
    """→ (frames, complete). `complete` False means coverage is truncated and NO clean verdict may
    be issued from it — the caller sends those to human review instead of approving."""
    frames = []
    dur = _duration(video_path)
    timeout = _timeout_for(dur)
    complete = True
    with tempfile.TemporaryDirectory() as d:
        cap = config.MAX_FRAMES
        if dur and dur > 0:
            # Even coverage: target between MIN_FRAMES and MAX_FRAMES, spread across the whole clip.
            target = min(config.MAX_FRAMES, max(config.MIN_FRAMES, round(dur * config.FPS_FLOOR)))
            even_fps = max(target / dur, 0.01)
            vf = f"fps={even_fps:.4f},{_SCALE}"
        else:
            target = config.MIN_FRAMES
            vf = f"fps={config.FPS_FLOOR},{_SCALE}"

        ok = _run_ffmpeg(video_path, vf, d, "even", cap, timeout, keyframes_only=True)
        got = _count(d, "even")
        # Keyframe decoding can come up short on a clip with very sparse keyframes (long GOP, or a
        # short clip with a single I-frame). Fall back to a full decode ONLY then, so we pay the
        # expensive path just for the files that need it.
        if got < min(config.MIN_FRAMES, target):
            print(f"[frames] only {got} keyframes; falling back to full decode", flush=True)
            ok = _run_ffmpeg(video_path, vf, d, "full", cap, timeout) and ok
            got = _count(d, "even") + _count(d, "full")
        if not ok:
            complete = False

        # Bonus: extra frames exactly at scene cuts (best-effort; may legitimately be empty). A
        # timeout here does NOT mark the scan incomplete — the even pass already guarantees
        # coverage, and this is additive.
        _run_ffmpeg(video_path, f"select='gt(scene,{config.SCENE_THRESHOLD})',{_SCALE}",
                    d, "scene", cap, timeout, keyframes_only=True)

        for fp in sorted(glob.glob(os.path.join(d, "*.jpg")))[: config.MAX_FRAMES]:
            try:
                im = Image.open(fp)
                im.load()
                frames.append(im.convert("RGB"))
            except Exception:
                continue

    # Even without a timeout, materially fewer frames than asked for means we did not see the clip
    # we thought we saw.
    if dur and dur > 0 and got < target * 0.8:
        print(f"[frames] coverage short: {got}/{target} frames", flush=True)
        complete = False
    return frames, complete
