"""Claude Haiku vision moderation — the accurate 'elegance' arbiter for VRE, and the second layer of
the hybrid gate (NudeNet free pre-filter → Claude only on the ambiguous band).

Unlike a body-part / skin-amount model, the vision LLM understands GARMENT TYPE and CONTEXT, so it
tells a business suit or a t-shirt+shorts from a bikini. Returns (verdict, reason):
verdict ∈ {"ALLOW","FLAG","BLOCK"}; "ERR" if the call failed (caller decides the fallback).
"""
import base64
import io
import json
import os
import time

import requests
from PIL import Image

_MODEL     = os.environ.get("VISION_MODEL", "claude-haiku-4-5-20251001")
_ENDPOINT  = "https://api.anthropic.com/v1/messages"
_MAX_EDGE  = int(os.environ.get("VISION_MAX_EDGE", "768"))   # 768px: enough detail, still cheap
_MAX_TOK   = int(os.environ.get("VISION_MAX_TOKENS", "80"))

# The bar: normal clothing of ANY style is fine (incl. tank tops, t-shirts, shorts, athletic/casual
# wear). Only swimwear / underwear / bare-torso / nudity / overtly sexual is out. This system prompt
# is IDENTICAL on every call, so it caches well (see README / prompt-caching note).
SYSTEM = (
    "You are the image moderator for VRE, a professional networking platform that stays free of "
    "nudity, underwear, and swimwear (like LinkedIn). Judge whether a user's uploaded photo belongs.\n\n"
    "ALLOW (the default for ANYONE in normal clothes): ordinary clothing of any style or fashion — "
    "suits, dresses, blouses, shirts, T-shirts, tank tops, sleeveless tops, hoodies, jeans, skirts "
    "(including short or pencil skirts), shorts (including athletic shorts), and casual, athletic, or "
    "fashionable outfits WORN WITH A TOP. Fitted/stylish clothing, visible arms and legs, and normal "
    "cleavage or necklines are all FINE. A tank top or sleeveless shirt counts as clothed. Also ALLOW "
    "non-people images (objects, products, scenery, logos, text).\n\n"
    "BLOCK only clear cases: swimwear (bikinis, one-piece swimsuits, speedos, or swim trunks/board "
    "shorts worn with NO shirt), lingerie, underwear or a bra worn as outerwear, a BARE shirtless "
    "torso (no top at all), any nudity or exposed breasts/genitals/buttocks, or overtly sexual/"
    "explicit content.\n\n"
    "FLAG only if genuinely borderline: a fully see-through top with nothing underneath, or a case you "
    "truly cannot resolve. Do NOT flag normal cleavage, fitted clothes, short skirts, or shorts.\n\n"
    "Decisive rule: if the person is wearing normal outer clothing (any top + bottoms), ALLOW — even "
    "fashionable, fitted, or showing some cleavage or leg. BLOCK ONLY swimwear, underwear/lingerie, a "
    "bare chest (no top), or nudity.\n\n"
    'Respond with ONLY compact JSON: {"verdict":"ALLOW|FLAG|BLOCK","reason":"<max 6 words>"}'
)

USER_TEXT = "Moderate this image. JSON only."


def _b64(img) -> str:
    if isinstance(img, str):
        img = Image.open(img)
    im = img.convert("RGB")
    im.thumbnail((_MAX_EDGE, _MAX_EDGE))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=85)
    return base64.b64encode(b.getvalue()).decode()


def moderate(img, api_key=None):
    """img: PIL.Image or path → (verdict, reason, {input_tokens, output_tokens}).

    Uses prompt caching on the (identical) system block so repeat calls only pay ~10% for it.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return "ERR", "no api key", {"input_tokens": 0, "output_tokens": 0}
    body = {
        "model": _MODEL,
        "max_tokens": _MAX_TOK,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(img)}},
            {"type": "text", "text": USER_TEXT},
        ]}],
    }
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    for attempt in range(4):
        try:
            r = requests.post(_ENDPOINT, headers=headers, data=json.dumps(body), timeout=60)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1)); continue
        if r.status_code == 200:
            d = r.json()
            txt = d["content"][0]["text"].strip()
            u = d.get("usage", {})
            usage = {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
                     "cache_read": u.get("cache_read_input_tokens", 0), "cache_write": u.get("cache_creation_input_tokens", 0)}
            try:
                j = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
                v = str(j.get("verdict", "")).upper()
                if v not in ("ALLOW", "FLAG", "BLOCK"):
                    v = "FLAG"
                return v, str(j.get("reason", ""))[:60], usage
            except Exception:
                return "FLAG", txt[:60], usage
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(2 * (attempt + 1)); continue
        return "ERR", f"http {r.status_code}", {"input_tokens": 0, "output_tokens": 0}
    return "ERR", "retries exhausted", {"input_tokens": 0, "output_tokens": 0}
