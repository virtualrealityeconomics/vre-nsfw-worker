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


def decide_image(img, api_key=None):
    scores = nsfw.detect_image(img)
    nn_status, nn_label, nn_score = config.verdict(scores)

    # Layer 1 — free, instant block on high-precision nudity/revealing signals.
    if nn_status == "rejected":
        return {"status": "rejected", "reason": f"nudity:{nn_label}", "layer": "nudenet",
                "scores": scores, "score": round(nn_score or 0.0, 4)}

    # If vision is off, fall back to the pure NudeNet gate (still blocks nudity; may miss bikinis).
    if not _vision_on():
        return {"status": nn_status, "reason": f"nudenet:{nn_label or 'clean'}", "layer": "nudenet-only",
                "scores": scores, "score": round(nn_score or 0.0, 4)}

    # Layer 2 — Claude vision for the elegance decision (the ambiguous majority).
    v, reason, usage = vision.moderate(img, api_key=api_key or config.ANTHROPIC_API_KEY)
    if v in _VMAP:
        return {"status": _VMAP[v], "reason": reason, "layer": "vision",
                "scores": scores, "score": round(nn_score or 0.0, 4), "usage": usage}

    # Vision errored → degrade. Default: hold for review (safe); fail-open approves instead.
    if config.VISION_FAIL_OPEN:
        fb = nn_status  # trust the free gate (approved/flagged)
    else:
        fb = "flagged" if nn_status == "approved" else nn_status
    return {"status": fb, "reason": f"vision_err:{reason}", "layer": "fallback",
            "scores": scores, "score": round(nn_score or 0.0, 4)}


def decide_images(imgs, api_key=None):
    """Worst verdict wins; short-circuit as soon as something is rejected (saves vision calls)."""
    worst = None
    for im in imgs:
        d = decide_image(im, api_key=api_key)
        if worst is None or _RANK[d["status"]] > _RANK[worst["status"]]:
            worst = d
        if worst["status"] == "rejected":
            break
    return worst or {"status": "approved", "reason": "no images", "layer": "none", "scores": {}, "score": 0.0}
