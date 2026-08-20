# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full-stack AI music generation app. Users upload or record audio; the backend transcribes it, analyzes pitch/chords/mood/key, and generates new melodies using trained PyTorch models (CVAE + IDDM-PPO). Output is MIDI or WAV, playable in the browser.

---

## Commands

### Frontend (Next.js — run from `frontend/`)

```bash
npm run dev      # Dev server on http://localhost:3000
npm run build    # Production build
npm run start    # Production server
```

### Backend (FastAPI — run from `backend/`)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Environment required: `PYTHONPATH=.` (set in `backend/.env`).

This alone is enough locally: the dev server also runs the transcription worker on a background thread (see "Async Transcription Job Workflow" below), so no second process is needed. Only run the worker separately if you've deliberately set `RUN_WORKER_IN_PROCESS=false` to mirror production:

```bash
python -m app.worker_main
```

### Tests (backend only)

```bash
cd backend
pip install -r requirements-dev.txt
pytest                                          # all tests
pytest tests/test_inference_variants.py        # inference + model loading
pytest tests/test_model.py -v                  # model tensor shapes
```

### Code Formatting

```bash
npx prettier --write .    # uses .prettierrc (100 char width, trailing commas, Tailwind plugin)
```

---

## Architecture

### Repository Layout

```text
music-generator-ai/
├── frontend/                    # Next.js 15 + React 19 + TypeScript + Tailwind CSS
├── backend/                     # FastAPI + PyTorch audio/ML pipeline
└── backend/melody_generation_ORDERED_FINAL_(1).ipynb  # Training notebook for CVAE + IDDM-PPO models
```

### Backend (`backend/app/`)

| File | Role |
| --- | --- |
| `main.py` | FastAPI app entry point — CORS, router registration |
| `inference.py` | Core logic: audio load → pitch/chord/mood/key analysis → VAE encode/decode → MIDI/WAV synthesis |
| `chord_utils.py` | Chord vocabulary (~170 entries: major/minor/aug/dim/5th across all 12 roots), tries to load from `web_model.pt` chord_vocab key, falls back to hardcoded list |
| `model/vae.py` | **Legacy** WebVAE — 2D depthwise-separable CNN VAE over mel spectrograms; not used in the active inference path |
| `model/colab_parity.py` | **Active** MelodyCVAE, MelStateEncoder, MelodyPPOActorCritic, TransitionDiscriminator, MINENetwork; used for all `/generate-variants` calls |
| `model/utils.py` | AudioProcessor, save_model/load_model helpers (target WebVAE, not used at inference time) |
| `model/weights/` | `.pth` checkpoint files — not committed (gitignored), loaded at runtime with graceful fallback |

**Audio pipeline in `inference.py`:**
librosa (load) → ffmpeg fallback for WebM/Opus → Basic Pitch (pitch detection) → librosa pyin fallback → music21 (chord/key) → CVAE encode/decode melody → pretty_midi + FluidSynth (optional WAV synthesis) → fallback sine-wave synthesizer

`_read_audio_bytes` (the shared decoder used by every audio-processing entry point) truncates decoded audio to `MAX_ANALYSIS_DURATION_SEC` (default 240s / 4 minutes, env-overridable) before any analysis runs, and probes the source file's header (via `soundfile`, no decode) to pass that cap straight into `librosa.load` so decode work itself is bounded rather than decoding the full upload before slicing. This constant used to be tied tightly to `GENERATION_TIMEOUT_SECONDS` (first 60s, then 30s) because transcription ran synchronously inside the HTTP request that timeout bounds, and Basic Pitch inference runs at roughly realtime speed on Render's free-tier CPU -- an uncapped clip reliably blew through both `GENERATION_TIMEOUT_SECONDS` and Render's own gateway timeout. Transcription now runs on the async job worker (see "Async Transcription Job Workflow" below), which isn't bound by any HTTP request lifetime, so this cap is decoupled from `GENERATION_TIMEOUT_SECONDS` entirely -- it now exists purely to bound worst-case per-job decode memory and worker occupancy. `_transcribe_and_mood`'s result (and the `/api/transcribe` job result) includes `source_duration_sec` (the full uploaded clip's duration) and `truncated: bool` alongside `duration_sec` (the duration actually analyzed), so the frontend can tell the user when their clip was cut short.

**Key directories at runtime:**

- `data/recordings/` — uploaded audio files (named `upload_{uuid}{ext}`)
- `data/generated/` — MIDI/WAV output
- `data/logs/` — JSON event logs per upload

### Two Model Architectures

There are **two separate model stacks** — only one is active in production:

| | WebVAE (`vae.py`) | MelodyCVAE + IDDM-PPO (`colab_parity.py`) |
| --- | --- | --- |
| Input | Mel spectrogram tensor `[B, 1, 128, 256]` | Token sequences (MIDI token IDs, Long tensor) |
| Weights file | `model/weights/web_model.pt` | `cvae_weights.pth` + `iddm_ppo_weights.pth` |
| Used by | Legacy `/generate` path (falls back gracefully if absent) | `/generate-variants`, `/transcribe`, active generate path |
| Training | — | `backend/notebook-new.ipynb` |

### Checkpoint Key Schemas

`cvae_weights.pth` must contain:

```python
{
  "model": <MelodyCVAE state_dict>,
  "cfg": {
    "vocab": 177, "emb_dim": 32, "hidden": 64, "latent_dim": 16,
    "n_moods": 3, "mel_bins": 80, "T_win": 16, "enc_dim": 64, "seq_len": 129
  }
}
```

`iddm_ppo_weights.pth` must contain:

```python
{
  "enc":  <MelStateEncoder state_dict>,
  "disc": <TransitionDiscriminator state_dict>,
  "ac":   <MelodyPPOActorCritic state_dict>,
  "mine": <MINENetwork state_dict>
}
```

Missing or mismatched keys raise HTTP 503 with a specific error message. The `cfg` dict from `cvae_weights.pth` drives all hyperparameters at inference time — if absent, hardcoded notebook defaults are used.

### MIDI Token Vocabulary (VOCAB_SIZE = 177)

| Range | Meaning |
| --- | --- |
| 0 – 127 | MIDI pitch (PITCH_OFFSET = 0) |
| 128 – 159 | Duration bins indexed into `DUR_BINS` (DUR_OFFSET = 128) |
| 160 – 175 | Tempo bins 40–200 BPM in steps of 10 (TEMPO_OFFSET = 160) |
| 176 | PAD token |

### Model Loading

`_CVAE_IDDM_BUNDLE` is a module-level singleton in `inference.py`, lazy-loaded on first request. A failed load is cached so every subsequent request returns the same 503 rather than retrying disk I/O. Reset by setting `inference._CVAE_IDDM_BUNDLE = None` (done in tests via monkeypatch).

### Async Transcription Job Workflow (`backend/app/jobs/`)

Transcription used to run Basic Pitch **synchronously inside the HTTP request** (`POST /api/transcribe`, handled directly in `inference.py`): the request thread blocked for up to `GENERATION_TIMEOUT_SECONDS`, and when that timeout fired, `asyncio.wait_for` only stopped the *caller* waiting — the underlying thread-pool worker kept running the inference to completion regardless (see `_run_generation`'s docstring), because Python has no way to forcibly cancel a running thread. That was a bad fit for Render: a slow clip tied up a request thread for the full budget, and a client retry after a timeout just queued identical work behind the still-running original. **That route has been retired** (its handler function is gone from `inference.py` entirely, not just unrouted) — `POST /api/transcribe` now means the async job-creation endpoint below. `run_basic_pitch`, the actual pipeline function it called, is untouched and is what the worker calls directly.

The job workflow decouples "accept the request" from "do the work":

```
POST /api/transcribe  →  create_transcription_job() → insert row (internal 'creating' state)
                           → write input bytes to object storage → mark_creation_ready()
                           (creating → queued) → push job_id onto the queue → 202 immediately

GET /api/transcribe/{id}  →  read current status/result/error from the store
                               ('creating' is reported to callers as "queued" -- never exposed)

Worker (separate thread in dev, separate process in production)
    → dequeue job_id → claim_job() (queued → processing, issues a fresh lease token)
    → read input bytes from storage → inference.run_basic_pitch() (same pipeline used before)
    → mark_completed(result, lease_token=...) or mark_failed(error, retry=..., lease_token=...)
       -- a write whose lease_token no longer matches the job's current one is discarded, not applied

Idle tick (no job to dequeue): reclaim_stale_processing_jobs(), get_ready_retry_job_ids(),
                                reconcile_stuck_creating_jobs() -- see below
```

| Module | Role |
| --- | --- |
| `jobs/routes.py` | `POST /transcribe`, `GET /transcribe/{id}` |
| `jobs/service.py` | `create_transcription_job()`; lazy singletons for the store/queue/storage adapters (same pattern as `_CVAE_IDDM_BUNDLE`) |
| `jobs/store.py` | `JobStore` protocol + `SQLJobStore` — job metadata (id, status, attempt count, lease token/expiry, backoff schedule, timestamps, result, error). One SQL implementation works against both `sqlite3` (dev/tests) and `psycopg`/Postgres (production); only the parameter placeholder differs |
| `jobs/queue.py` | `JobQueue` protocol — `InProcessJobQueue` (in-memory, dev/tests/single-instance) or `RedisJobQueue` (BRPOP/LPUSH, production) |
| `jobs/storage.py` | `ObjectStorage` protocol — `LocalFilesystemStorage` (dev/tests) or `S3CompatibleStorage` (boto3; AWS S3 or, via `endpoint_url`, Cloudflare R2) for the job's input audio bytes, and (when configured) a mirror of worker-generated MIDI/WAV output |
| `jobs/worker.py` | `process_one_job` / `run_worker_loop` — the actual claim → run pipeline → update status logic, independent of how it's hosted |
| `worker_main.py` | Standalone production entrypoint: `python -m app.worker_main` |

**Why a queue *and* a store, not just one:** the store is the durable source of truth (a job's status survives a worker crash or restart); the queue is just a fast handoff so a worker isn't polling the database in a tight loop for new work. `InProcessJobQueue` is pure in-memory and doesn't survive a restart on its own — `run_worker_loop` covers this by re-enqueueing anything already `queued` in the store at startup, and separately reclaims jobs stuck in `processing` via `reclaim_stale_processing_jobs`, checked on every idle poll.

**Atomic idempotency, safe across multiple web instances:** `create_transcription_job` keys on `upload_id` when the client referenced an existing upload, or a SHA-256 of the raw bytes otherwise. Dedup is enforced by a **partial unique database index** (`idempotency_key` where `status != 'failed'`) plus `create_job`'s `INSERT ... ON CONFLICT (idempotency_key) WHERE status != 'failed' DO NOTHING`, not an in-process lock -- a Python-level lock only protects one process, and Postgres/SQLite (3.24+) both support the identical `ON CONFLICT ... WHERE` syntax against a partial unique index, so the same `SQLJobStore` code is correct whether there's one web instance or several. A client retrying a job-creation call (network blip, cold-start failure) can't spin up a second, redundant transcription -- this is what makes it safe for the frontend to retry job creation unconditionally, unlike the old synchronous transcription route this replaced. A permanently-`failed` job is excluded from the index, so a genuine retry after a permanent failure still gets a fresh row for the same key.

**Reliable job creation despite three non-atomic steps:** a DB insert, an object-storage write, and a queue push can't be made transactional across two different systems. Instead the row starts in the internal `creating` state (never exposed via the API -- `Job.to_status_payload()` reports it as `"queued"`) and only flips to `queued` after the storage write succeeds. If the creating process dies in between, `reconcile_stuck_creating_jobs` (part of every worker's idle tick, so it runs regardless of whether the web service and worker are the same process) finds rows stuck in `creating` past a short threshold, checks whether the input bytes actually made it to storage, and either releases the job to `queued` (bytes exist -- just the status transition/enqueue was lost) or fails it outright (bytes were never written -- nothing to run).

**Lease fencing -- stale workers can't overwrite newer results:** `claim_job` issues a fresh `lease_token` and `lease_expires_at` (queued → processing). `mark_completed`/`mark_failed` only take effect if the caller's `lease_token` still matches the job's current one (`UPDATE ... WHERE id = ? AND lease_token = ?`); a write with a stale token matches zero rows and is discarded (logged, not treated as an error). `reclaim_stale_processing_jobs` clears the lease in the *same* `UPDATE` that requeues an expired job, so it's atomic with respect to the original worker's own completion write -- whichever of the two the database serializes first wins, and the other one's `WHERE` clause simply matches nothing. This does **not** stop the original worker from continuing to run Basic Pitch after its lease is reclaimed (Python can't force-kill a running thread -- the same limitation `GENERATION_TIMEOUT_SECONDS` already has); it only guarantees that whichever attempt finishes first wins cleanly instead of the two silently corrupting each other's result.

**Retries are actually redelivered, with backoff:** a pipeline failure that raises `HTTPException` with a 4xx status (bad/undecodable audio, empty payload -- see `_read_audio_bytes`) is treated as permanent, since the input won't decode differently next time; a 5xx `HTTPException` or any other exception is retried up to the job's `max_attempts` (default 3). A retried job doesn't go straight back into the queue -- `mark_failed(retry=True)` sets `next_attempt_at` to an exponential backoff (2s, 4s, 8s, ... capped at 60s) and leaves it `queued`; the worker loop's idle tick calls `get_ready_retry_job_ids()` once that delay has elapsed and enqueues it then. (An earlier version of this moved the row back to `queued` without ever re-enqueueing it, silently orphaning every retried job until the worker process restarted -- covered by `test_run_worker_loop_automatically_retries_a_transient_failure`, which drives a real `run_worker_loop` rather than calling `process_one_job` directly, since the bug was specifically in the loop-level redelivery path.) Stored error messages are always short, pre-existing user-facing strings (an `HTTPException.detail`) or the fixed string `"Transcription failed unexpectedly"` — never a raw exception message or traceback.

**Shared generated-file storage:** `inference.run_basic_pitch` (via `_save_bytes`) always writes the generated MIDI/WAV to local disk, regardless of which process calls it. That would be fine if the same process that ran the pipeline also served `/api/download`, but the worker calling it may be a separate Render service from the web process. When object storage is configured (`JOB_STORAGE_BUCKET` set), the worker mirrors both files into it under a `generated/` key prefix right after `run_basic_pitch` returns; `download_generated_file` (`/api/download/{filename}`) falls back to object storage when the file isn't on local disk. `inference.py` has no import-time dependency on `jobs/` -- the fallback uses the same lazy-import pattern already used elsewhere in this codebase. In local dev (no bucket configured, `LocalFilesystemStorage.is_shared = False`), nothing is mirrored and behavior is unchanged from before this existed.

**Why not MongoDB:** job state here is small, relational, and finite-state (`creating → queued → processing → {completed, failed}`, plus retry/lease bookkeeping) — exactly what a SQL table with a couple of indexes is for. A document store wouldn't remove the need for the queue or the lease fencing (which is what actually solves the request-timeout and stale-overwrite problems) and would just be an extra moving part with no matching shape of data.

**`RUN_WORKER_IN_PROCESS`** (default `true`) controls whether `main.py`'s lifespan starts the worker loop on a background thread inside the web process **and** whether it eagerly warms up (loads) Basic Pitch at startup -- both are gated on this flag, so `RUN_WORKER_IN_PROCESS=false` means the web process does neither. Locally this means `uvicorn app.main:app --reload` alone is enough. **Production must set this to `false` on the web service** and run `python -m app.worker_main` as a separate Render Background Worker service instead — leaving it on in production is exactly the "web process spins up a fire-and-forget transcription thread" problem this workflow exists to remove, just moved one layer down. (An earlier version of this had a residual caveat here: a legacy synchronous transcribe route still called `run_basic_pitch` directly regardless of this flag. That route is gone now -- nothing outside the worker calls `run_basic_pitch` anymore, so `RUN_WORKER_IN_PROCESS=false` genuinely means the web process never runs the pipeline.)

### Frontend (`frontend/app/`)

Next.js App Router. Most backend communication is proxied through Next.js API routes in `frontend/app/api/` via `_lib/backend.ts`. The exceptions are the two calls that carry the full audio file — upload and transcribe — which the browser sends directly to the FastAPI backend (see below), because Vercel's serverless functions have a hard, non-configurable ~4.5MB request body limit that large audio files can exceed. Any other route that ends up carrying a full file body (not just an `id`/`filename` reference) needs the same treatment, or it will hit the same limit.

| Route | Purpose |
| --- | --- |
| `/` (`page.tsx`) | Landing page -- hero, pipeline overview, links to `/analyse` and the training notebook |
| `/analyse` (`analyse/page.tsx`) | Upload/record audio, display transcription results |
| `/choose-progression` | Select a chord progression |
| `/generate-variants` | Generate melody variants |
| `/listen-progressions` | Playback and analysis |

**Direct-to-backend calls** (bypass the Next.js proxy, see above), both built via `frontend/app/lib/backendUrl.ts`'s `getPublicBackendApiUrl`:

- `frontend/app/lib/upload.ts` posts the audio file straight to `POST {NEXT_PUBLIC_BACKEND_URL}/api/upload`, returning `{id, filename}`.
- `frontend/app/lib/transcribeJob.ts`'s `createTranscribeJob` posts to `POST {NEXT_PUBLIC_BACKEND_URL}/api/transcribe`, returning `{job_id, status}` immediately (202). It takes an optional `{id, filename}` upload reference; `analyse/page.tsx` always has one by the time it calls this (the file is uploaded first either way), so it sends `upload_id`/`filename` form fields and references the already-uploaded copy instead of sending the full file a second time -- the raw-file path only exists as a fallback for a caller with no prior upload. `getTranscribeJob`/`pollTranscribeJob` then poll `GET {NEXT_PUBLIC_BACKEND_URL}/api/transcribe/{job_id}` with bounded exponential backoff until the job completes or fails — see "Async Transcription Job Workflow" above.

There are no `frontend/app/api/upload/route.ts` or `frontend/app/api/transcribe*/route.ts` — all of these were removed when these fixes were made (the job endpoints carry the same full-file-body concern as `/upload` did, and inherit the same direct-to-backend treatment).

**API route proxies** (each forwards to FastAPI):

- `POST /api/generate` — generate melody
- `POST /api/generate-progression` — generate chord progressions
- `POST /api/generate-variants` — create melody variants
- `GET /api/chords` — list available chords
- `GET /api/download/[filename]` — stream generated file

All backend routes are canonicalized under `/api/*` (served by `inference.router`, mounted with that prefix in `main.py`), except `/api/upload`, `/model-info`, `/process/`, and `/health`, which are registered directly on the FastAPI app.

**Backend URL resolution in `_lib/backend.ts`** (server-side only, precedence order):

1. `BACKEND_BASE_URL` env var
2. `NEXT_PUBLIC_BACKEND_URL` env var
3. `http://127.0.0.1:8000` (default)

### Request Flow

```text
Browser → Next.js page → Next.js API route → FastAPI backend → inference.py → model → response
```

Upload flow: browser → `POST {NEXT_PUBLIC_BACKEND_URL}/api/upload` directly (not proxied) → stores `upload_{uuid}{ext}` → returns `{id, filename}` → `id` is passed through the normal Next.js proxy to `/api/generate` to reference the stored file. `/analyse` also immediately kicks off transcription of the same file via `POST {NEXT_PUBLIC_BACKEND_URL}/api/transcribe` directly (not proxied, for the same file-size reason), then polls `GET .../api/transcribe/{job_id}` until the job completes to show analysis results.

---

## Key Configuration

- **Frontend env:** `frontend/.env.local` — set `NEXT_PUBLIC_BACKEND_URL` (or `BACKEND_BASE_URL` for server-only) to backend URL. `NEXT_PUBLIC_BACKEND_URL` is inlined into the client bundle at build time and must be a publicly reachable backend URL in production, since the browser calls it directly for audio upload
- **Backend env:** `backend/.env` — `PYTHONPATH=.`
- **Audio params override:** `model/weights/audio_params.json` — if present, overrides `DEFAULT_AUDIO_CFG` for mel spectrogram processing
- **Soundfont:** `app/soundfonts/GeneralUser-GS.sf2` or `SOUNDFONT_PATH` env var — used by FluidSynth for WAV synthesis
- **CORS:** `main.py` reads `CORS_ALLOWED_ORIGINS` (comma-separated, defaults to `http://localhost:3000`) — must include the deployed frontend origin in production, since the browser calls the backend directly for uploads
- **PyTorch:** CPU wheels only, sourced from `https://download.pytorch.org/whl/cpu`
- **ffmpeg:** Required at runtime as fallback decoder for WebM/Opus browser recordings that librosa cannot decode natively
- **FluidSynth:** Optional system package; without it, MIDI synthesis falls back to a sine-wave synthesizer
- **Supported audio formats:** `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.webm`
- **Default sample rate:** 22,050 Hz
- **Max analyzed audio duration:** `MAX_ANALYSIS_DURATION_SEC` env var, defaults to 240s (4 minutes) — clips longer than this are truncated before analysis; see `_read_audio_bytes` in Architecture above. This used to be tied to `GENERATION_TIMEOUT_SECONDS` (first 60s, then 30s) because transcription ran synchronously inside a request that constant bounds; now that transcription runs on the async job worker (unbound by any HTTP request lifetime), this cap exists purely to bound worst-case per-job memory and worker occupancy, not to dodge a timeout. `jobs/worker.py`'s `DEFAULT_LEASE_SECONDS` (900s) is sized with real margin above this value, so raising this further would also mean raising that.
- **Generation timeout:** `GENERATION_TIMEOUT_SECONDS` env var, defaults to 60s. Measured in production, Render's own platform-level request timeout sits close to 100s, and CPU starvation on the free tier can add ~20-30s of lag before our own timeout error is actually delivered — so this needs real margin below Render's limit, not just to be lower than it, or Render's raw CORS-header-less 502 wins the race instead of our clean JSON error. This now applies only to the *other* synchronous routes (`/generate-variants`, etc) — transcription no longer runs inside a request this bounds, since its old synchronous route was retired in favor of the async job workflow. `pollTranscribeJob`'s ~15 minute client-side poll budget is a separate, unrelated number: how long the *browser* waits across possibly-many polls, sized with the same margin above `MAX_ANALYSIS_DURATION_SEC` as the worker's lease. The frontend retries a failed `/api/upload` or `/api/transcribe` (job creation) call once, since a cold Render instance can fail the very first request outright while it's still waking up, and a retry right after is effectively always fast. Job creation is safe to retry unconditionally (no 504 carve-out needed, unlike the old synchronous route): it's just a fast enqueue, and the backend's idempotency key (see "Async Transcription Job Workflow" above) dedupes a retried creation instead of spinning up a redundant job.
- **Async transcription job workflow** (see "Async Transcription Job Workflow" above) — all optional; omitting all of them gives a fully working local setup (SQLite + in-process queue + local disk + in-process worker thread):
  - `RUN_WORKER_IN_PROCESS` — `true` (default) runs the worker on a background thread inside the web process (dev). Set to `false` in production and run `python -m app.worker_main` as a separate service.
  - `DATABASE_URL` — Postgres DSN for job metadata in production (falls back to SQLite if unset).
  - `JOB_DB_PATH` — SQLite file path override for local dev (defaults to `backend/data/jobs.db`); ignored if `DATABASE_URL` is set.
  - `REDIS_URL` — Redis connection URL for the production job queue (falls back to an in-process queue if unset).
  - `JOB_STORAGE_BUCKET` — S3/R2 bucket name for job input audio in production (falls back to local disk under `backend/data/job_storage/` if unset).
  - `JOB_STORAGE_ENDPOINT_URL` — custom S3-compatible endpoint (e.g. Cloudflare R2's account endpoint); omit for real AWS S3.
  - `JOB_STORAGE_REGION`, `JOB_STORAGE_ACCESS_KEY_ID`, `JOB_STORAGE_SECRET_ACCESS_KEY` — credentials/region for `JOB_STORAGE_BUCKET`; omit to fall back to boto3's standard credential chain (env vars, instance role, etc).

---

## Deployment

- **Frontend:** Vercel (`frontend/vercel.json`)
- **Backend web service:** Docker (`backend/Dockerfile`) — Python 3.10-slim, exposes port 8000. Set `RUN_WORKER_IN_PROCESS=false` here in production (see "Async Transcription Job Workflow" above) — otherwise this service still runs transcription inference on its own thread, which is the exact problem the job workflow exists to remove.
- **Backend worker service:** a second Render **Background Worker** service, same repo/image as the web service (same `backend/Dockerfile`), with its start command overridden in the Render dashboard to `python -m app.worker_main` instead of the Dockerfile's default `uvicorn` CMD. No `render.yaml` exists in this repo (Render config here is dashboard-configured, per existing convention) — both services are created and configured directly in the dashboard, sharing the same environment variables (`DATABASE_URL`, `REDIS_URL`, `JOB_STORAGE_*`, model checkpoint paths, etc) so they see the same job store/queue/storage.
- **Job metadata (production):** a Render Postgres instance; set its connection string as `DATABASE_URL` on both the web and worker services.
- **Job queue (production):** Redis — a Render Key Value instance (or any Redis-compatible service); set its URL as `REDIS_URL` on both services.
- **Job input audio storage (production):** an S3-compatible bucket (Cloudflare R2 or AWS S3); set `JOB_STORAGE_BUCKET` (+ `JOB_STORAGE_ENDPOINT_URL` for R2, `JOB_STORAGE_REGION`, and credentials) on both services. This is required, not optional, once the worker runs as a separate Render service from the web service — they don't share a disk, so the worker can't read input audio the web service wrote to local disk.
- **Frontend Docker:** `frontend/Dockerfile` — node:18-alpine, runs `npm run dev`
