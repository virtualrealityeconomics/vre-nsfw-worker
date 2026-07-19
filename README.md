# vre-nsfw-worker

Headless NSFW moderation poller. Reads Postgres directly, fetches media from the public R2 URL,
classifies with **Falconsai/nsfw_image_detection** (Apache-2.0) via ONNX Runtime (CPU, model baked
into the image), and writes `moderationStatus`. No inbound HTTP except a secret-gated `/metrics`.

## Flow (agreed 2026-07-20)
- **Image / text post** — published immediately (`moderationStatus='approved'`, optimistic). The image
  loop scans un-scanned ones (`approved` + `moderatedAt IS NULL`). NSFW → flip to `rejected`/`flagged`
  (hides it) + quarantine the R2 bytes. SFW → stamp `moderatedAt` (stays public). Fast, seamless.
- **Video** (Posts-page *or* Videos-page) — created `pending` (hidden). SFW → set `Video.moderationStatus
  ='approved'`; the existing Vultr transcode worker claims **only approved** videos, transcodes, then
  flips the Post public + notifies. NSFW → `Video`+`Post` `rejected` + quarantine.
- **Orphan video post** (video enroll failed → no Video row) — scanned directly via `Post.videoUrl`;
  SFW → `approved` (plays raw MP4); NSFW → `rejected`.

## Reliability
- Claims are lease-based (`moderationLockedAt`) + `FOR UPDATE SKIP LOCKED`. **Scale by running more
  replicas** — each grabs different rows, a crash self-heals (stale lease → re-claimable).
- Two threads: a fast **image** loop and a slower **video** loop, so a long video scan never stalls
  image publishes.
- Scan errors retry to `NSFW_MAX_ATTEMPTS`, then a terminal `error` state (hidden, surfaced to admin) —
  a legit upload is never invisibly stuck. Unscannable video (ffmpeg yields nothing) fails closed
  (retry, never approve unseen).

## Deploy
Railway (Docker), no public domain (private networking + `/metrics` on `$PORT`). Same pattern as
`vre-face-svc`. The model is exported to ONNX and baked at build (no runtime download). Env: see
`.env.example` — needs `DIRECT_URL`, the `R2_*` creds, and `METRICS_SECRET`.

## ⚠️ Depends on (NOT built yet)
- Schema: `moderationStatus` / `moderatedAt` / `moderationScore` / `moderationLabels` /
  `moderationAttempts` / `moderationLockedAt` on `Post` and `Video` (manual delta).
- vre.pro: create routes set the initial status + the hidden-video UX (toast/backoffice); feed filters.
- Vultr video worker: claim gated on `moderationStatus='approved'` + the "ready" notification.
