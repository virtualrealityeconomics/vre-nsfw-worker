# vre-nsfw-worker

Headless image/video moderation poller for an elegant, professional social platform. It reads
Postgres directly, fetches media from a public R2 URL, decides `approved` / `flagged` / `rejected`,
and quarantines blocked bytes. No inbound HTTP except a secret-gated `/metrics`.

## How it decides — a hybrid gate

**Images** run a two-layer gate:

1. **NudeNet v3 (free, instant).** A per-body-part detector (via the [`nudenet`](https://pypi.org/project/nudenet/)
   package, ONNX Runtime, CPU). If it detects explicit nudity or clearly revealing content
   (exposed genitals/breasts/buttocks, bare midriff, thong), the image is **rejected immediately** —
   no second call needed. This catches the obvious cases for $0.
2. **Claude vision (the elegance decision).** Everything else goes to a small vision model
   (`claude-haiku-4-5`) that understands *garment type and context* — the difference between a
   business suit and a bikini, a tank top and a bare chest — which a body-part detector cannot. It
   returns `ALLOW` / `FLAG` / `BLOCK` with a short reason.

**The standard:** block only the bright lines — swimwear/bikini, lingerie, underwear, a bare
shirtless torso, nudity, and overtly sexual content. Normal clothing of any style (suits, dresses,
t-shirts, tank tops, shorts, fashion/athletic wear, ordinary cleavage) is allowed.

**Videos** use **NudeNet only** (no vision calls — per-frame vision would be costly). Frames are
sampled with ffmpeg (a guaranteed floor of evenly-spaced frames plus scene-change frames), scored
per label, and aggregated by max — so a single explicit frame fails the whole clip.

## Reliability & scaling

- Claims are lease-based (`moderationLockedAt`) + `FOR UPDATE SKIP LOCKED` — **scale by running more
  replicas**; each grabs different rows and a crash self-heals (stale lease → re-claimable).
- Two threads: a fast **image** loop and a slower **video** loop, so a long video scan never stalls
  image publishes.
- Scan errors retry to `NSFW_MAX_ATTEMPTS`, then a terminal `error` state (hidden, surfaced to
  admin) — a legit upload is never invisibly stuck. Unscannable video fails closed (retry, never
  approve unseen). If the vision API errors, the image is held for review (configurable).

## Run locally

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env            # fill in DIRECT_URL, R2_*, ANTHROPIC_API_KEY, METRICS_SECRET
./.venv/bin/python main.py
```

There is also a local tester (`serve_test.py`) that serves an upload page and runs the real gate on
a single image or video — drag a file in and see the per-body-part breakdown, the Claude verdict,
and how it was decided. It reads `ANTHROPIC_API_KEY` from the env or a repo-local `.env.local`.

## Deploy

Docker (see `Dockerfile`), no public domain — private networking + `/metrics` on `$PORT`. The
NudeNet model ships inside the `nudenet` wheel (no runtime download). Env: see `.env.example`.

## Licensing / NOTICE

- This worker's own code is provided under the terms in `LICENSE`.
- The **`nudenet` Python package** is MIT-licensed and runs purely on ONNX Runtime.
- The bundled NudeNet v3 detector weights (`320n.onnx`) are derived from **Ultralytics YOLOv8**,
  which is licensed **AGPL-3.0**. This repository is kept **open-source** and operates at arm's
  length from the closed-source application it serves (they communicate only through the database),
  so the application is not a derivative work. No Ultralytics code runs at inference time.
- Image moderation calls the **Anthropic Claude API** (a paid, hosted service).

If you fork or deploy this, review these obligations for your own use.
