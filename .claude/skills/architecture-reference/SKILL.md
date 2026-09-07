---
description: Full architecture reference for melody-generator-ai — non-functional requirements, backend/frontend architecture, async job workflow internals, model checkpoint schemas, key configuration, deployment, and the production-readiness checklist. Load this when a task needs implementation detail beyond CLAUDE.md's spine.
---

# Architecture Reference

Moved out of CLAUDE.md on 2026-08-24 to keep the file that loads every turn short. This is
the detailed, fast-changing reference material CLAUDE.md points to — treat it as living
documentation, not a fixed contract: three of its "gap" bullets went stale in the very
session this was split out (marked **Fixed** below), so verify anything load-bearing
against the actual code before relying on it.

---

## Non-Functional Requirements

Scoped to a solo-maintainer, free/cheap-tier budget — not enterprise SLAs — but these are requirements, not aspirations. Gaps called out below should be closed before onboarding real users beyond a small trusted group. See "Production Readiness" below for the concrete provisioning checklist these requirements motivate.

### Availability

- No formal SLA. Target: the app should be *usable*. The free-tier backend is kept warm by `.github/workflows/keep-warm.yml` (pings `/health` every 10 min) so it doesn't idle-suspend, but a deploy or the rare crash still restarts it — that must degrade gracefully (visible loading state + the frontend's retry-once on job creation, not a hung request or a raw 502). Under load, only one heavy audio task runs at a time (see "Cost constraints"); the excess is queued (transcription) or 429'd (generation) with a clear message, never silently stalled.
- A restart must never silently lose an in-flight job. This now holds in production: the job stack (Neon Postgres + Render Redis + Backblaze B2) went live and was verified end-to-end on 2026-08-24 (see "Production Readiness" below) — job durability comes from those three external services, not from the worker's own process, so it holds even with the worker still running in-process on the web service (see below). Until they were wired up, the dev-only fallback stack (see "Async Transcription Job Workflow" below) lost any `queued`/`processing` job on restart with no user-visible explanation; that's now only a concern for local dev, not production.

### Error handling

- Every user-facing failure must surface a specific, actionable message — never a raw stack trace or an unhandled 500. `HTTPException` with a fixed `detail` string is the existing convention (see `_read_audio_bytes`, job failure messages in "Async Transcription Job Workflow"); keep following it for new endpoints.
- FastAPI's default behavior for an *unhandled* (non-`HTTPException`) exception includes the exception message in the response body — an information leak once real users can trigger it. **Fixed:** `main.py` now registers `handle_unexpected_error`, a catch-all `@app.exception_handler(Exception)` that logs the real error via `logger.exception` and returns a sanitized `{"detail": "An unexpected error occurred. Please try again."}` with a 500 status.
- The async job retry policy (permanent failure on 4xx, retried with backoff on 5xx/unexpected, capped at `max_attempts`) is the existing model for anything that talks to a flaky external system — follow the same 4xx-permanent / 5xx-retryable split for new failure-prone work rather than inventing a new policy per endpoint.

### Observability

- Structured, correlated logging is a gap: logging today is plain `logger.info`/`logger.exception` calls with no request ID or job ID consistently threaded through, so tracing one user's request across web → queue → worker in production logs is manual. At minimum, job-related log lines should carry `job_id` end-to-end.
- No error tracking (e.g. Sentry) is wired up on either service — an unhandled exception in production is visible only in Render's log stream, which isn't practical to monitor continuously as a solo maintainer.
- `/health` is a liveness check only (process is up), not readiness (DB/queue/storage reachable). Once the production job stack is live, `/health` can report healthy while jobs actually can't run because Redis or Postgres is unreachable.
- No uptime monitoring or alerting is configured on either Render service.

### Security

- No authentication or authorization exists anywhere in the app — every route is fully public. That's an accepted tradeoff for a demo with no user accounts or private data; rate limiting (below) is the actual defense against abuse in its place.
- **Fixed — rate limiting is now live** (`backend/app/rate_limit.py`: an in-memory, per-IP, single-process limiter — see its own docstring on why that stops being correct once there's more than one web instance, at which point it'd need to move to a shared store like the existing Redis). `POST /api/transcribe` and `POST /api/generate-variants` both carry a `rate_limiter(...)` route dependency, default 5 requests / 300s per IP, configurable via `TRANSCRIBE_RATE_LIMIT`/`TRANSCRIBE_RATE_WINDOW_SECONDS` and `GENERATE_VARIANTS_RATE_LIMIT`/`GENERATE_VARIANTS_RATE_WINDOW_SECONDS`.
- **Fixed — request body size cap is now live** (`backend/app/request_limits.py`'s `MaxBodySizeMiddleware`, mounted in `main.py` after `CORSMiddleware` so it runs outermost and rejects before any route body-read). Caps every route, not just `/api/upload`, at `MAX_UPLOAD_BYTES` (default 50MB, env-overridable) — a 413 with `RequestBodyTooLarge`/`handle_request_body_too_large`, using `Content-Length` when present and a streaming byte-count fallback when it isn't.
- CORS is already origin-restricted via `CORS_ALLOWED_ORIGINS` (not `*`) — keep this as-is; don't widen it for convenience.
- Secrets (`DATABASE_URL`, `REDIS_URL`, `JOB_STORAGE_*` credentials) are dashboard-configured per Render service, not committed — keep it that way. There's no documented rotation policy, which is fine at this scale but worth knowing if a credential ever needs to be treated as compromised.

### Cost constraints

**The backend runs on Render's free web-service tier (512MB / 0.1 CPU) and this is a permanent decision — a deliberate cost tradeoff for a solo portfolio project, reconfirmed 2026-09-07 after a full memory-hardening pass. Do not propose paying to lift it.** Vercel's hobby tier ($0) hosts the frontend. The job stack (Neon Postgres / Render Key Value / Backblaze B2) is also all free tier.

The 512MB ceiling is the sharpest constraint. What was done to make it work, and what it costs:

- **Transcription memory is now length-independent.** `TRANSCRIBE_CHUNKED=true` transcribes the whole upload in overlapping 30s windows (`_transcribe_full_chunked` + `_merge_chunk_notes`), so the posteriorgram never exceeds one chunk. Measured: a 5-minute clip peaks at ~400–430MB, flat across chunks. See `docs/PLAN-full-transcription-split.md`.
- **`torch` is lazy-imported.** `inference.py` and `colab_parity`'s model classes don't load torch at module import — only the generation paths do, and only after the weights-exist check passes. The transcription path never pays torch's ~150MB. Generation with no weights installed (the current deployment) is a ~1ms 503 that never touches torch. (Verified by measurement in the Phase B audit, not just by design.)
- **Import surface audited (Phase B, 2026-09-07).** Idle floor is ~290MB after warm-up. `music21` was removed (−35MB boot RSS, 239 submodules) — it did key detection that `_key_from_histogram` (Krumhansl-Schmuckler) already does as its fallback. `matplotlib` / `scikit-learn` / `tensorflow` are **not** imported by any path (older docs claiming "sklearn in the floor" were wrong). `librosa` / `scipy` / `numba` / `tflite_runtime` / `mir_eval` / `resampy` are pulled eagerly by `basic_pitch.inference` + warm-up and can't be deferred without moving cost into the transcription peak. `TRANSCRIBE_CHUNK_SEC` 30→15 was measured — no peak benefit (per-chunk working set ~11MB either way), kept at 30. `torch` in the Docker image (~200MB disk) is unused dead weight; removing it is a "generation permanently off" decision, deferred to the user.
- **Only one heavy audio task runs at a time, enforced in code.** `HEAVY_WORK_LOCK` (in `inference.py`) serializes the transcribe worker against every generation route. A generation request that can't get the lock returns **429 immediately** (`_run_generation` acquires non-blocking) with a "try again in a moment" message — *not* a block that piles up threads which all OOM when they finally run. The transcribe worker, being a single thread with jobs queued behind it in the store, instead waits `WORKER_HEAVY_WORK_WAIT_SEC` (180s) then raises so the job is re-queued. `_release_memory_to_os()` (glibc `malloc_trim`) runs after every heavy task.
- Ephemeral disk → the external job stack is why job durability doesn't depend on the process; anything on local disk (`backend/data/`) is lost on restart/redeploy.
- Basic Pitch runs at ~real-time speed on this CPU — with the one-at-a-time serialization, a burst of users queues visibly. Rate limiting (live) and the serialization are both cost/latency controls, not just abuse-prevention.
- The instance doesn't idle-suspend only because `.github/workflows/keep-warm.yml` pings `/health` every 10 min. `healthCheckPath` on the Render service is still unset (a manual dashboard fix — set it to `/health`).

Historical (the 2026-08-24 measurement that drove all of the above): before the lazy-torch split and chunking, a *single* transcription climbed to ~530MB of the 536MB limit, and a *second* concurrent request reliably OOM-killed the instance mid-run. That failure mode is now designed out — one task at a time, torch deferred, transcription chunked.

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
| `main.py` | FastAPI app entry point — CORS, body-size middleware, catch-all exception handler, router registration |
| `inference.py` | Core logic: audio load → pitch/chord/mood/key analysis → VAE encode/decode → MIDI/WAV synthesis |
| `rate_limit.py` | In-memory per-IP rate limiter (`rate_limiter(...)` FastAPI dependency), applied to `/api/transcribe` and `/api/generate-variants` |
| `request_limits.py` | `MaxBodySizeMiddleware` — rejects oversized request bodies before they're read into memory |
| `chord_utils.py` | Chord vocabulary (~170 entries: major/minor/aug/dim/5th across all 12 roots), tries to load from `web_model.pt` chord_vocab key, falls back to hardcoded list |
| `model/vae.py` | **Legacy** WebVAE — 2D depthwise-separable CNN VAE over mel spectrograms; not used in the active inference path |
| `model/colab_parity.py` | **Active** MelodyCVAE, MelStateEncoder, MelodyPPOActorCritic, TransitionDiscriminator, MINENetwork; used for all `/generate-variants` calls |
| `model/utils.py` | AudioProcessor, save_model/load_model helpers (target WebVAE, not used at inference time) |
| `model/weights/` | `.pth` checkpoint files — not committed (gitignored), loaded at runtime with graceful fallback |

**Audio pipeline in `inference.py`:**
librosa (load) → ffmpeg fallback for WebM/Opus → Basic Pitch (pitch detection) → librosa pyin fallback → chord/key analysis (librosa chroma + `_key_from_histogram`, a Krumhansl-Schmuckler profile match — music21 was removed 2026-09-07, Phase B import audit, for −35 MB boot RSS) → CVAE encode/decode melody → pretty_midi + FluidSynth (optional WAV synthesis) → fallback sine-wave synthesizer

**Two transcription paths, gated on `TRANSCRIBE_CHUNKED` (production: `true`).**

- **Chunked (`_transcribe_and_mood_chunked` → `_transcribe_full_chunked`):** the whole upload is transcribed in overlapping 30s / 4s-overlap windows (`TRANSCRIBE_CHUNK_SEC` / `TRANSCRIBE_OVERLAP_SEC`), decoded one window at a time via `librosa.load(offset=, duration=)`, so peak memory doesn't grow with clip length. `_merge_chunk_notes` welds notes across boundaries — see `docs/PLAN-full-transcription-split.md` for the algorithm (the risky part). Output MIDI is full-length, `truncated` always `false`. `MAX_UPLOAD_DURATION_SEC` (600s / 10 min) is the hard cap. The mood/key/BPM/chord analysis runs on the first `MAX_ANALYSIS_DURATION_SEC` (60s) only, reusing the merged notes + one librosa pass (librosa-only, no second Basic Pitch call). The worker renews the job lease per chunk (`JobStore.heartbeat`) so a long job isn't reclaimed mid-run.
- **Single-pass (`_transcribe_and_mood`, legacy — plan step 4 deletes it):** `_read_audio_bytes` truncates decode to `MAX_ANALYSIS_DURATION_SEC` via a `soundfile` header probe. The chunked path falls back to this only when duration can't be probed (webm mic recordings).

Both results carry `source_duration_sec`, `duration_sec`, `truncated`, and (chunked) `n_chunks`.

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

> **Dev-only fallback stack.** Every adapter below (job store, queue, object storage, worker) has a same-process, zero-config fallback — SQLite, `InProcessJobQueue`, local disk, and an in-process worker thread — so `uvicorn app.main:app --reload` alone works locally with no external services. That fallback is for dev and tests only. It must not be what production traffic runs on: Render's free-tier web service has ephemeral disk (SQLite and local files are lost on every restart/redeploy), the in-process queue and worker don't survive a restart or work across separate processes, and a single process is a single point of failure for both serving requests and running inference. See "Non-Functional Requirements" above and "Production Readiness" below.

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
| `jobs/storage.py` | `ObjectStorage` protocol — `LocalFilesystemStorage` (dev/tests) or `S3CompatibleStorage` (boto3; AWS S3 or, via `endpoint_url`, an S3-compatible provider like Backblaze B2) for the job's input audio bytes, and (when configured) a mirror of worker-generated MIDI/WAV output |
| `jobs/worker.py` | `process_one_job` / `run_worker_loop` — the actual claim → run pipeline → update status logic, independent of how it's hosted |
| `worker_main.py` | Standalone production entrypoint: `python -m app.worker_main` |

**Why a queue *and* a store, not just one:** the store is the durable source of truth (a job's status survives a worker crash or restart); the queue is just a fast handoff so a worker isn't polling the database in a tight loop for new work. `InProcessJobQueue` is pure in-memory and doesn't survive a restart on its own — `run_worker_loop` covers this by re-enqueueing anything already `queued` in the store at startup, and separately reclaims jobs stuck in `processing` via `reclaim_stale_processing_jobs`, checked on every idle poll.

**Atomic idempotency, safe across multiple web instances:** `create_transcription_job` keys on `upload_id` when the client referenced an existing upload, or a SHA-256 of the raw bytes otherwise. Dedup is enforced by a **partial unique database index** (`idempotency_key` where `status != 'failed'`) plus `create_job`'s `INSERT ... ON CONFLICT (idempotency_key) WHERE status != 'failed' DO NOTHING`, not an in-process lock -- a Python-level lock only protects one process, and Postgres/SQLite (3.24+) both support the identical `ON CONFLICT ... WHERE` syntax against a partial unique index, so the same `SQLJobStore` code is correct whether there's one web instance or several. A client retrying a job-creation call (network blip, cold-start failure) can't spin up a second, redundant transcription -- this is what makes it safe for the frontend to retry job creation unconditionally, unlike the old synchronous transcription route this replaced. A permanently-`failed` job is excluded from the index, so a genuine retry after a permanent failure still gets a fresh row for the same key.

**Reliable job creation despite three non-atomic steps:** a DB insert, an object-storage write, and a queue push can't be made transactional across two different systems. Instead the row starts in the internal `creating` state (never exposed via the API -- `Job.to_status_payload()` reports it as `"queued"`) and only flips to `queued` after the storage write succeeds. If the creating process dies in between, `reconcile_stuck_creating_jobs` (part of every worker's idle tick, so it runs regardless of whether the web service and worker are the same process) finds rows stuck in `creating` past a short threshold, checks whether the input bytes actually made it to storage, and either releases the job to `queued` (bytes exist -- just the status transition/enqueue was lost) or fails it outright (bytes were never written -- nothing to run).

**Lease fencing -- stale workers can't overwrite newer results:** `claim_job` issues a fresh `lease_token` and `lease_expires_at` (queued → processing). `mark_completed`/`mark_failed` only take effect if the caller's `lease_token` still matches the job's current one (`UPDATE ... WHERE id = ? AND lease_token = ?`); a write with a stale token matches zero rows and is discarded (logged, not treated as an error). `reclaim_stale_processing_jobs` clears the lease in the *same* `UPDATE` that requeues an expired job, so it's atomic with respect to the original worker's own completion write -- whichever of the two the database serializes first wins, and the other one's `WHERE` clause simply matches nothing. This does **not** stop the original worker from continuing to run Basic Pitch after its lease is reclaimed (Python can't force-kill a running thread -- the same limitation `GENERATION_TIMEOUT_SECONDS` already has); it only guarantees that whichever attempt finishes first wins cleanly instead of the two silently corrupting each other's result.

**Retries are actually redelivered, with backoff:** a pipeline failure that raises `HTTPException` with a 4xx status (bad/undecodable audio, empty payload -- see `_read_audio_bytes`) is treated as permanent, since the input won't decode differently next time; a 5xx `HTTPException` or any other exception is retried up to the job's `max_attempts` (default 3). A retried job doesn't go straight back into the queue -- `mark_failed(retry=True)` sets `next_attempt_at` to an exponential backoff (2s, 4s, 8s, ... capped at 60s) and leaves it `queued`; the worker loop's idle tick calls `get_ready_retry_job_ids()` once that delay has elapsed and enqueues it then. (An earlier version of this moved the row back to `queued` without ever re-enqueueing it, silently orphaning every retried job until the worker process restarted -- covered by `test_run_worker_loop_automatically_retries_a_transient_failure`, which drives a real `run_worker_loop` rather than calling `process_one_job` directly, since the bug was specifically in the loop-level redelivery path.) Stored error messages are always short, pre-existing user-facing strings (an `HTTPException.detail`) or the fixed string `"Transcription failed unexpectedly"` — never a raw exception message or traceback.

**`RedisJobQueue` must disable the client's own socket read timeout.** `redis-py`'s default `socket_timeout` is 5s (`redis._defaults.DEFAULT_SOCKET_TIMEOUT`) -- and `worker.DEFAULT_POLL_TIMEOUT_SECONDS` (what `dequeue()` rounds up into BRPOP's server-side blocking timeout argument) also defaults to 5.0s. If the client is constructed without overriding `socket_timeout`, the two end up numerically identical, and the client's own read deadline races BRPOP's "block up to N seconds, then return nil" promise on every empty poll -- discovered in production on 2026-08-24 as `dequeue()` intermittently raising `redis.exceptions.TimeoutError: Timeout reading from socket` instead of cleanly returning `None`. Combined with `run_worker_loop`'s dequeue call not being wrapped in a try/except (unlike `process_one_job`'s call a few lines below it, which already guards against this class of failure), a single one of these errors silently killed the entire worker thread -- it's a daemon thread with nothing supervising it, so the web process kept serving requests fine while nothing ever dequeued again. Fixed by (a) wrapping `dequeue()` in try/except-and-continue, matching the existing `process_one_job` pattern, and (b) constructing the Redis client with `socket_timeout=None` so BRPOP's own timeout argument is what actually bounds the wait. `socket_connect_timeout` is untouched (still redis-py's 5s default) so a genuinely unreachable Redis still fails fast on connect. If `RedisJobQueue`'s constructor or `DEFAULT_POLL_TIMEOUT_SECONDS` are ever changed, keep `socket_timeout=None` -- reintroducing any finite client-side socket_timeout reopens this race regardless of its exact value, since it only takes one being `<=` the BRPOP timeout under real network latency.

**Shared generated-file storage:** `inference.run_basic_pitch` (via `_save_bytes`) always writes the generated MIDI/WAV to local disk, regardless of which process calls it. That would be fine if the same process that ran the pipeline also served `/api/download`, but the worker calling it may be a separate Render service from the web process. When object storage is configured (`JOB_STORAGE_BUCKET` set), the worker mirrors both files into it under a `generated/` key prefix right after `run_basic_pitch` returns; `download_generated_file` (`/api/download/{filename}`) falls back to object storage when the file isn't on local disk. `inference.py` has no import-time dependency on `jobs/` -- the fallback uses the same lazy-import pattern already used elsewhere in this codebase. In local dev (no bucket configured, `LocalFilesystemStorage.is_shared = False`), nothing is mirrored and behavior is unchanged from before this existed.

**Why not MongoDB:** job state here is small, relational, and finite-state (`creating → queued → processing → {completed, failed}`, plus retry/lease bookkeeping) — exactly what a SQL table with a couple of indexes is for. A document store wouldn't remove the need for the queue or the lease fencing (which is what actually solves the request-timeout and stale-overwrite problems) and would just be an extra moving part with no matching shape of data.

**`RUN_WORKER_IN_PROCESS`** (default `true`) controls whether `main.py`'s lifespan starts the worker loop on a background thread inside the web process **and** whether it eagerly warms up (loads) Basic Pitch at startup -- both are gated on this flag, so `RUN_WORKER_IN_PROCESS=false` means the web process does neither. Locally, `uvicorn app.main:app --reload` alone is enough. **Production also runs it `true`** — one free-tier service doing both the API and the worker. `HEAVY_WORK_LOCK` (see "Cost constraints") makes the in-process arrangement safe by serializing the worker's inference against the API's generation routes: only one heavy task at a time, a second one gets a 429. Splitting a dedicated Background Worker out (`RUN_WORKER_IN_PROCESS=false` on the web service + `python -m app.worker_main` as its own service) is the cleaner shape and a real option for later; nothing outside the worker calls `run_basic_pitch`, so it genuinely means the web process never runs the pipeline.

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
- **Analysis clip window:** `MAX_ANALYSIS_DURATION_SEC` env var, default 60s. With `TRANSCRIBE_CHUNKED` this no longer truncates the transcription (the whole upload is transcribed) — it's the window the mood/key/BPM/chord analysis runs on. `MAX_UPLOAD_DURATION_SEC` (600s) is the hard upload cap; `TRANSCRIBE_CHUNK_SEC` (30) / `TRANSCRIBE_OVERLAP_SEC` (4) are the chunk geometry. Plan steps 2–3 rename this to a "default clip" constant and make the bounds a UI choice. `jobs/worker.py` renews the lease per chunk (`heartbeat`, `HEARTBEAT_LEASE_SECONDS` 180s) so a long chunked job doesn't outlive `DEFAULT_LEASE_SECONDS` (900s).
- **Heavy-work lock wait:** `WORKER_HEAVY_WORK_WAIT_SEC` env var, default 180s — how long the transcribe worker waits for an in-flight generation before raising (job re-queued). The generation routes don't wait at all — they 429 immediately.
- **Generation timeout:** `GENERATION_TIMEOUT_SECONDS` env var, defaults to 60s. Measured in production, Render's own platform-level request timeout sits close to 100s, and CPU starvation on the free tier can add ~20-30s of lag before our own timeout error is actually delivered — so this needs real margin below Render's limit, not just to be lower than it, or Render's raw CORS-header-less 502 wins the race instead of our clean JSON error. This now applies only to the *other* synchronous routes (`/generate-variants`, etc) — transcription no longer runs inside a request this bounds, since its old synchronous route was retired in favor of the async job workflow. `pollTranscribeJob`'s ~15 minute client-side poll budget is a separate, unrelated number: how long the *browser* waits across possibly-many polls, sized with the same margin above `MAX_ANALYSIS_DURATION_SEC` as the worker's lease. The frontend retries a failed `/api/upload` or `/api/transcribe` (job creation) call once, since a cold Render instance can fail the very first request outright while it's still waking up, and a retry right after is effectively always fast. Job creation is safe to retry unconditionally (no 504 carve-out needed, unlike the old synchronous route): it's just a fast enqueue, and the backend's idempotency key (see "Async Transcription Job Workflow" above) dedupes a retried creation instead of spinning up a redundant job.
- **Request body cap:** `MAX_UPLOAD_BYTES` env var, defaults to 50MB (52428800) — enforced by `MaxBodySizeMiddleware` on every route, not just `/api/upload`. See "Security" above.
- **Rate limiting:** `TRANSCRIBE_RATE_LIMIT`/`TRANSCRIBE_RATE_WINDOW_SECONDS` and `GENERATE_VARIANTS_RATE_LIMIT`/`GENERATE_VARIANTS_RATE_WINDOW_SECONDS`, both defaulting to 5 requests / 300 seconds per IP — see "Security" above.
- **Async transcription job workflow** (see "Async Transcription Job Workflow" above) — all optional in dev; omitting all of them gives the **dev-only** fallback setup (SQLite + in-process queue + local disk + in-process worker thread) — not fit for real-user production traffic (see "Non-Functional Requirements" and "Production Readiness"):
  - `RUN_WORKER_IN_PROCESS` — `true` (default, and what production runs). `false` + `python -m app.worker_main` as a separate service is the split-out option, not deployed.
  - `DATABASE_URL` — Postgres DSN for job metadata in production (falls back to SQLite if unset).
  - `JOB_DB_PATH` — SQLite file path override for local dev (defaults to `backend/data/jobs.db`); ignored if `DATABASE_URL` is set.
  - `REDIS_URL` — Redis connection URL for the production job queue (falls back to an in-process queue if unset).
  - `JOB_STORAGE_BUCKET` — S3-compatible bucket name for job input audio in production (falls back to local disk under `backend/data/job_storage/` if unset).
  - `JOB_STORAGE_ENDPOINT_URL` — custom S3-compatible endpoint (e.g. Backblaze B2's S3-compatible endpoint, `https://s3.<region>.backblazeb2.com`); omit for real AWS S3.
  - `JOB_STORAGE_REGION`, `JOB_STORAGE_ACCESS_KEY_ID`, `JOB_STORAGE_SECRET_ACCESS_KEY` — credentials/region for `JOB_STORAGE_BUCKET`; omit to fall back to boto3's standard credential chain (env vars, instance role, etc).

---

## Deployment

- **Frontend:** Vercel (`frontend/vercel.json`)
- **Backend web service:** Docker (`backend/Dockerfile`) — Python 3.10-slim, port 8000. Render **free** tier, Frankfurt. Runs `RUN_WORKER_IN_PROCESS=true` (API + worker in one service) and `TRANSCRIBE_CHUNKED=true`; `HEAVY_WORK_LOCK` keeps that safe on 512MB (one heavy task at a time). Split the worker out later if wanted — see "Async Transcription Job Workflow" above.
- **Backend worker service:** a second Render **Background Worker** service, same repo/image as the web service (same `backend/Dockerfile`), with its start command overridden in the Render dashboard to `python -m app.worker_main` instead of the Dockerfile's default `uvicorn` CMD. No `render.yaml` exists in this repo (Render config here is dashboard-configured, per existing convention) — both services are created and configured directly in the dashboard, sharing the same environment variables (`DATABASE_URL`, `REDIS_URL`, `JOB_STORAGE_*`, model checkpoint paths, etc) so they see the same job store/queue/storage.
- **Job metadata (production):** Postgres — currently **Neon** (external managed Postgres, free tier), not a Render Postgres instance; `SQLJobStore` just needs a `psycopg`-compatible DSN, so either works identically. Set the connection string as `DATABASE_URL` on both the web and worker services. Neon connection strings need `?sslmode=require`.
- **Job queue (production):** Redis — a Render Key Value instance (free tier); set its URL as `REDIS_URL` on both services. The free Key Value plan has **no external connection option and no dashboard Shell** on the web service either — verifying Redis connectivity outside of an actual deploy isn't possible on this plan; see the `RedisJobQueue` socket_timeout note above for a bug this limitation made harder to diagnose.
- **Job input audio storage (production):** an S3-compatible bucket — currently **Backblaze B2**, chosen over Cloudflare R2 on cost and provisioning grounds: B2's per-GB storage rate (~$0.006/GB) runs roughly 60% below R2's ($0.015/GB) once usage exceeds the free tier, its free tier doesn't require a payment method on file, and R2's headline differentiator (unconditional zero-egress) isn't decisive at this project's volume, since uploaded/generated audio stays well within B2's free egress allowance (3x average monthly storage). Set `JOB_STORAGE_BUCKET` (+ `JOB_STORAGE_ENDPOINT_URL` for B2's S3-compatible endpoint, `JOB_STORAGE_REGION`, and credentials) on both services. This is required, not optional, once the worker runs as a separate Render service from the web service — they don't share a disk, so the worker can't read input audio the web service wrote to local disk.
- **Frontend Docker:** `frontend/Dockerfile` — node:18-alpine, runs `npm run dev`

---

## Production Readiness

What's still missing to run this for real users, beyond a thesis/demo. Checked directly against the live Render account (one workspace) as of 2026-08-24. See "Non-Functional Requirements" above for why each of these matters.

### Infrastructure

- **Postgres** — **live and verified** (2026-08-24). Job metadata (`DATABASE_URL`) is a Neon Postgres instance (free tier, external — see "Deployment" above), not the SQLite fallback. Verified end-to-end: a job created via the real `/api/transcribe` API walked `queued → processing → completed` with its row tracked in Neon throughout. (Along the way, found and fixed a real bug this surfaced: `SQLJobStore._init_schema` never committed its `CREATE TABLE` before a migration loop that fails-and-rolls-back on Postgres specifically, discarding the uncommitted table — see `backend/app/jobs/store.py`. SQLite never hit this, so the existing test suite didn't catch it.)
- **Redis** — **live and verified** (2026-08-24). The job queue (`REDIS_URL`) is a Render Key Value instance (free tier), not the in-process fallback. Verified end-to-end after fixing a real bug it surfaced: see the `RedisJobQueue` socket_timeout note above (in "Async Transcription Job Workflow") for the full story — a `redis-py` default racing `BRPOP`'s own timeout was silently killing the worker thread on every occurrence.
- **Object storage** — **live and verified** (2026-08-24). Job input audio and generated-file mirroring (`JOB_STORAGE_BUCKET`) go to a Backblaze B2 bucket (free tier), not local disk. Verified both directions: a job's input bytes landed at `transcribe-jobs/{job_id}/input.wav`, and the worker's generated MIDI/WAV output was mirrored to `generated/` under the exact filenames in the job's result payload.
- **Worker service — not provisioned, deliberately.** Only one Render service exists (`melody-generator-ai`, the free-tier web service); `RUN_WORKER_IN_PROCESS` stays `true`. On the free tier the in-process arrangement is made safe by `HEAVY_WORK_LOCK` (one heavy task at a time; a second generation request 429s, a queued transcription waits) rather than by the split. Chunked transcription + lazy torch keep a single transcription under the 512MB ceiling. Splitting a dedicated Background Worker out stays a clean option for later — not a fix that's overdue.

### CI/CD

- CI exists (`.github/workflows/ci.yml`): frontend runs format-check/typecheck/test/build, backend runs `pytest`. Real coverage — keep it required on `main`.
- Gaps: no backend lint/type-check step (no `ruff`/`mypy`, unlike the frontend's typecheck), no frontend lint step either (no ESLint config exists in `frontend/` at all yet — see this repo's CLAUDE.md "Open questions"), no Docker build verification (the backend `Dockerfile` could break without CI catching it), no dependency/secret scanning, and no deploy gate — Render currently auto-deploys on every push to `main` (`autoDeploy: yes`) independent of whether CI passed. Worth deciding whether deploys should instead trigger off CI success rather than the raw push event, so a red CI run can't ship.

### Monitoring & observability

- No error tracking (Sentry or equivalent) on either frontend or backend.
- No uptime monitoring or alerting on either Render service.
- No structured/correlated logging (see "Non-Functional Requirements" above).
- `/health` is liveness-only, not readiness — it doesn't check DB/queue/storage reachability.

### Security

- No auth still — accepted tradeoff for now, see "Non-Functional Requirements" above.
- Rate limiting and the request body size cap are both now fixed (see "Security" under "Non-Functional Requirements" above) — the two bullets that used to be here.

### Data retention

- `cleanup_stale_files`/`run_periodic_cleanup` (`inference.py`) already deletes stale files from local `data/recordings/` and `data/generated/` on a schedule (`DATA_RETENTION_HOURS`, default 24h) — but this only covers local disk. Now that B2 object storage is actually live (see "Production Readiness" above), uploaded audio and job input bytes under `transcribe-jobs/{job_id}/input{ext}` (plus mirrored output under `generated/`) have **no equivalent retention policy** and will accumulate (and cost money) indefinitely — this is now a live, accumulating gap, not a future one. Needs either a B2 bucket lifecycle rule or an equivalent application-level cleanup — and, separately, a stated retention/deletion policy for user-uploaded audio, since real users means real (if informal) privacy expectations.

### Anything else spotted

- **FluidSynth WAV synthesis is currently falling back to the sine-wave synthesizer in production** — observed directly in logs during the 2026-08-24 verification: `WARNING:app.inference:FluidSynth render failed: fluidsynth() was called but pyfluidsynth is not installed.` The `backend/Dockerfile` doesn't currently install the `pyfluidsynth` Python package (or its system-level FluidSynth dependency), so every generated WAV is the lower-quality fallback rather than real soundfont synthesis, silently. Not investigated further as part of this pass — flagging since it directly affects output quality for every real user, not just a dev-environment gap.
- No frontend error boundary/crash reporting wired to a reporting service — `react-error-boundary` is already a frontend dependency and at least one component (`error-toast`) has test coverage, but there's no top-level boundary + Sentry-style capture yet, so a client-side exception in `/analyse` or elsewhere can still fall through to React's default error UI rather than a clean, reported fallback.
- No load or cost testing has been done against the free/starter tiers this is meant to run on — worth a deliberate small-scale test (a handful of concurrent transcriptions) before wider release, since Basic Pitch runs at roughly real-time speed on free-tier CPU and a burst of real users will queue up visibly.
