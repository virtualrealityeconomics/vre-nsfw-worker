"""Hybrid moderation gate — the decision the worker makes for an image.

  Layer 1  NudeNet (free, instant): if the body-part gate says REJECT (explicit nudity / bare midriff
           / thong / exposed buttocks — the high-precision signals that never fired on elegant photos
           in testing), block immediately for $0. No Claude call.
  Layer 2  Claude Haiku vision: everything else → the accurate elegance judgment. The vision LLM
           understands garment TYPE and CONTEXT (business-suit vs bikini, tank-top vs shirtless,
           bare midriff, micro-shorts, heavy cleavage) — which body-part models fundamentally cannot.

decide_image(img)  -> {status, reason, layer, scores, score, usage?}
decide_images(list) -> worst verdict across a post's images (any REJECT blocks the post; short-circuits
                        so a NudeNet reject on image 1 never pays for vision on the rest).

Fallback: if vision errors, degrade to the NudeNet verdict (never hard-fail; still blocks nudity).
Set NSFW_VISION_FAIL_OPEN=1 to approve-on-error instead of holding for review.
"""
import config
import nsfw
import vision

_RANK = {"approved": 0, "flagged": 1, "rejected": 2}
_VMAP = {"ALLOW": "approved", "FLAG": "flagged", "BLOCK": "rejected"}


def _vision_on():
    return config.VISION_ENABLED and bool(config.ANTHROPIC_API_KEY)


def decide_image(img, api_key=None, free_block=True, scores=None):
    """One image → verdict. `free_block=True` (images): NudeNet auto-blocks the whole reject band.
    `free_block=False` (video frames): ONLY explicit nudity auto-blocks (config.HARD_NUDITY) — the soft
    'elegance' band is routed to vision so a NudeNet false-positive (squirrel→BELLY) can be rescued.
    `scores` lets a caller reuse a NudeNet result it already computed (skips re-inference)."""
    scores = scores if scores is not None else nsfw.detect_image(img)
    nn_status, nn_label, nn_score = config.verdict(scores)
    h_status, h_label, h_score = config.hard_hit(scores)

    # 1. Explicit nudity is NON-overridable (images AND videos): a hard-class BLOCK always rejects.
    if h_status == "rejected":
        return {"status": "rejected", "reason": f"nudity:{h_label}", "layer": "nudenet",
                "scores": scores, "score": round(h_score or 0.0, 4)}
    # 2. Images keep the original free-block over the whole reject band (bare-belly / thong / etc.).
    if free_block and nn_status == "rejected":
        return {"status": "rejected", "reason": f"nudity:{nn_label}", "layer": "nudenet",
                "scores": scores, "score": round(nn_score or 0.0, 4)}
    # 3. VIDEOS: low-confidence explicit nudity → hold for human review; a 320px vision must NOT approve it.
    if not free_block and h_status == "flagged":
        return {"status": "flagged", "reason": f"lowconf:{h_label}", "layer": "nudenet",
                "scores": scores, "score": round(h_score or 0.0, 4)}
    # 4. Vision arbitrates the elegance band (the ambiguous majority) — or NudeNet fallback if off.
    if not _vision_on():
        return {"status": nn_status, "reason": f"nudenet:{nn_label or 'clean'}", "layer": "nudenet-only",
                "scores": scores, "score": round(nn_score or 0.0, 4)}
    v, reason, usage = vision.moderate(img, api_key=api_key or config.ANTHROPIC_API_KEY)
    if v in _VMAP:
        return {"status": _VMAP[v], "reason": reason, "layer": "vision",
                "scores": scores, "score": round(nn_score or 0.0, 4), "usage": usage}
    # Vision errored → degrade fail-closed: a frame NudeNet already called 'rejected'/'flagged' STAYS so;
    # an 'approved' one is held for review. NSFW never auto-approves on a vision error.
    if config.VISION_FAIL_OPEN:
        fb = nn_status
    else:
        fb = "flagged" if nn_status == "approved" else nn_status
    return {"status": fb, "reason": f"vision_err:{reason}", "layer": "fallback",
            "scores": scores, "score": round(nn_score or 0.0, 4)}


def decide_images(imgs, api_key=None, free_block=True):
    """Worst verdict wins; short-circuit as soon as something is rejected (saves vision calls)."""
    worst = None
    for im in imgs:
        d = decide_image(im, api_key=api_key, free_block=free_block)
        if worst is None or _RANK[d["status"]] > _RANK[worst["status"]]:
            worst = d
        if worst["status"] == "rejected":
            break
    return worst or {"status": "approved", "reason": "no images", "layer": "none", "scores": {}, "score": 0.0}


def _ahash(img):
    """64-bit average hash — cheap near-duplicate detector so 3 stills of the SAME shot aren't all sent
    to vision (they'd be a wasted, identical call)."""
    im = img.convert("L").resize((8, 8))
    px = list(im.getdata())
    avg = (sum(px) / len(px)) if px else 0
    bits = 0
    for i, p in enumerate(px):
        if p >= avg:
            bits |= (1 << i)
    return bits


def _dedupe(scored):
    """scored = [(img, scores, tripping_score)] pre-sorted by score desc → drop near-duplicate frames."""
    kept, hashes = [], []
    for item in scored:
        h = _ahash(item[0])
        if any(bin(h ^ kh).count("1") <= config.VIDEO_DEDUPE_HAMMING for kh in hashes):
            continue
        hashes.append(h)
        kept.append(item)
    return kept


def decide_video(agg, suspicious, api_key=None):
    """VIDEO gate. NudeNet is a pre-filter: EVERY suspicious frame is checked for explicit nudity for FREE
    (a hard BLOCK on ANY frame rejects the clip — never evictable by a higher-scoring false positive). Only
    the ELEGANCE band — clearly suspicious (≥ VIDEO_VISION_MIN), deduped, capped to VIDEO_VISION_FRAMES —
    pays a vision call. Sub-threshold soft signals are treated as benign.
      agg        : {label: max across all frames}   (persisted for logs)
      suspicious : [(frame_img, frame_scores)]  frames whose NudeNet verdict != 'approved'.
    """
    if not suspicious:
        return {"status": "approved", "reason": "clean", "layer": "nudenet",
                "scores": agg, "score": 0.0, "vision_frames": 0}

    hard_flag = None
    soft = []  # (img, scores, tripping_score) — elegance-band, worth vision
    for (img, sc) in suspicious:
        h_status, h_label, h_score = config.hard_hit(sc)
        if h_status == "rejected":
            return {"status": "rejected", "reason": f"nudity:{h_label}", "layer": "nudenet",
                    "scores": agg, "score": round(h_score or 0.0, 4), "vision_frames": 0}
        if h_status == "flagged":
            if hard_flag is None:
                hard_flag = (h_label, h_score)
            continue  # low-confidence nudity → review, not a vision candidate
        nn_status, _, ts = config.verdict(sc)
        # A soft-BLOCK frame (confidently revealing — bikini / bare-midriff / shirtless) ALWAYS gets a
        # vision call: its block threshold can sit BELOW VIDEO_VISION_MIN, so the MIN gate must not
        # silently approve it. The soft-FLAG low-confidence noise tail stays MIN-gated (cost control).
        if nn_status == "rejected" or (ts or 0.0) >= config.VIDEO_VISION_MIN:
            soft.append((img, sc, ts or 0.0))

    soft.sort(key=lambda t: t[2], reverse=True)
    soft = _dedupe(soft)[: config.VIDEO_VISION_FRAMES]

    vd, calls = None, 0
    for (img, sc, _ts) in soft:
        calls += 1
        d = decide_image(img, api_key=api_key, free_block=False, scores=sc)
        if vd is None or _RANK[d["status"]] > _RANK[vd["status"]]:
            vd = d
        if vd["status"] == "rejected":
            break

    # Combine the vision verdict with any held hard-flag (worst wins). Sub-MIN soft signals = benign.
    status, reason, layer, score = "approved", "clean-ish", "nudenet", 0.0
    if vd is not None:
        status, reason, layer, score = vd["status"], vd["reason"], vd["layer"], vd["score"]
    if hard_flag and _RANK["flagged"] > _RANK[status]:
        status, reason, layer, score = "flagged", f"lowconf:{hard_flag[0]}", "nudenet", hard_flag[1]
    return {"status": status, "reason": reason, "layer": layer,
            "scores": agg, "score": round(score or 0.0, 4), "vision_frames": calls}
