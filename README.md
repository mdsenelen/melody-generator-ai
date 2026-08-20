# Melody Generator AI

Melody Generator AI turns a piece of audio — uploaded or recorded in the browser — into new,
playable melody variants. Upload or record → the backend transcribes the pitch, chords, key and
mood → a trained CVAE + IDDM-PPO model generates melody variants and chord progressions → you
preview and download the result as MIDI or WAV.

```
Record / upload audio
        │
        ▼
 Transcription (pitch, chords, key, mood)
        │
        ▼
  Choose a chord progression
        │
        ▼
 Generate melody variants (CVAE + IDDM-PPO)
        │
        ▼
 Listen, compare, download MIDI/WAV
```

## Architecture

```
Browser
  │  (never talks to FastAPI directly)
  ▼
Next.js pages (frontend/app)
  │
  ▼
Next.js API routes (frontend/app/api/*) ──proxy──▶ FastAPI backend (backend/app)
                                                        │
                                                        ▼
                                          inference.py: audio load (librosa,
                                          ffmpeg fallback) → Basic Pitch /
                                          pyin pitch detection → music21
                                          chord & key analysis → CVAE encode/
                                          decode → pretty_midi + FluidSynth
                                          (sine-wave fallback) → MIDI/WAV
```

The frontend never calls the FastAPI server directly — every request goes through a Next.js API
route in `frontend/app/api/`, which proxies to the backend URL resolved in
`frontend/app/api/_lib/backend.ts`.

The backend has two model stacks in the codebase, but only one is active at inference time:

| | `model/vae.py` (WebVAE) | `model/colab_parity.py` (MelodyCVAE + IDDM-PPO) |
| --- | --- | --- |
| Status | Legacy, not used in the live inference path | **Active** — used by `/generate-variants`, `/transcribe`, and the main generate path |
| Input | Mel spectrogram tensor | MIDI token sequences (177-token vocabulary) |
| Weights | `model/weights/web_model.pt` | `cvae_weights.pth` + `iddm_ppo_weights.pth` |
| Trained by | — | `backend/melody_generation_ORDERED_FINAL_(1).ipynb` |

Model weight checkpoints are **not committed** to this repository (they're large binary
artifacts). Without them, generation endpoints return a `503` with a specific "which checkpoint
key is missing" error rather than failing silently — see `CLAUDE.md` for the exact schema.

## Tech stack

- **Frontend**: Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, Jest
- **Backend**: FastAPI, PyTorch (CPU wheels), librosa, Basic Pitch, music21, pretty_midi
- **Audio synthesis**: FluidSynth (optional; falls back to a sine-wave synthesizer if unavailable)
- **Decode fallback**: ffmpeg, for WebM/Opus recordings librosa can't read natively

## Getting started

### Prerequisites

- Node.js 18+
- Python 3.10
- `ffmpeg` on your `PATH` (required as a decode fallback for browser recordings)
- FluidSynth + a SoundFont (optional — enables real WAV synthesis instead of the sine-wave
  fallback); see `app/soundfonts/GeneralUser-GS.sf2` or set `SOUNDFONT_PATH`

### Backend

```bash
cd backend
cp .env.example .env          # sets PYTHONPATH=.
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

This alone is enough — transcription runs on a background worker thread inside this same process by default (see "Async Transcription Job Workflow" in `CLAUDE.md`). Only run the worker as a separate process if you've set `RUN_WORKER_IN_PROCESS=false`:

```bash
python -m app.worker_main
```

### Frontend

```bash
cd frontend
cp .env.local.example .env.local   # points at the local backend
npm install
npm run dev                        # http://localhost:3000
```

## Testing

```bash
# Backend
cd backend
pip install -r requirements-dev.txt
pytest

# Frontend
cd frontend
npm test            # unit tests (Jest)
npm run typecheck   # tsc --noEmit
npm run format:check  # prettier --check .
```

CI (`.github/workflows/ci.yml`) runs install, format check, typecheck, tests, and build for the
frontend, and `pytest` for the backend, on every push and pull request.

## Environment variables

| File | Variable | Purpose |
| --- | --- | --- |
| `backend/.env` | `PYTHONPATH` | Set to `.` so `app.*` imports resolve when running uvicorn |
| `backend/.env` (optional) | `SOUNDFONT_PATH` | Overrides the default FluidSynth SoundFont path |
| `backend/.env` (optional) | `GENERATION_TIMEOUT_SECONDS` | Wall-clock timeout for a single synchronous generation request, e.g. `/generate-variants` (default `60`) — needs real margin below Render's own ~100s platform timeout, see `CLAUDE.md`. Transcription no longer runs inside a request this bounds; see the async job workflow below |
| `backend/.env` (optional) | `MAX_ANALYSIS_DURATION_SEC` | Caps how much of an uploaded clip is decoded/analyzed (default `240`, 4 minutes) — longer clips are truncated, decode itself is bounded to this window rather than decoding the full upload first. No longer tied to `GENERATION_TIMEOUT_SECONDS`; bounds worst-case worker memory/occupancy instead, see `CLAUDE.md` |
| `backend/.env` (optional) | `DATA_RETENTION_HOURS` | How long uploaded/generated files are kept before periodic cleanup deletes them (default `24`; `0` disables cleanup) |
| `backend/.env` (optional) | `DATA_CLEANUP_INTERVAL_SECONDS` | How often the background cleanup pass runs (default `3600`) |
| `backend/.env` (optional) | `CORS_ALLOWED_ORIGINS` | Comma-separated list of allowed origins (default `http://localhost:3000`) — must include the deployed frontend origin, since the browser calls the backend directly for uploads |
| `backend/.env` (optional) | `RUN_WORKER_IN_PROCESS` | Runs the transcription worker on a background thread inside this process (default `true`) — set `false` in production and run `python -m app.worker_main` as its own service instead. See "Async Transcription Job Workflow" in `CLAUDE.md` |
| `backend/.env` (optional) | `DATABASE_URL` | Postgres DSN for job metadata in production; falls back to a local SQLite file if unset |
| `backend/.env` (optional) | `REDIS_URL` | Redis URL for the production job queue; falls back to an in-process queue if unset |
| `backend/.env` (optional) | `JOB_STORAGE_BUCKET`, `JOB_STORAGE_ENDPOINT_URL`, `JOB_STORAGE_REGION`, `JOB_STORAGE_ACCESS_KEY_ID`, `JOB_STORAGE_SECRET_ACCESS_KEY` | S3/R2 bucket (+ credentials) for job input audio in production; falls back to local disk if unset. Required once the worker runs as a separate service from the web process, since they don't share a disk |
| `frontend/.env.local` | `BACKEND_BASE_URL` | Server-side backend URL used by Next.js API routes |
| `frontend/.env.local` | `NEXT_PUBLIC_BACKEND_URL` | Backend URL inlined into the client bundle; the browser posts audio uploads and transcription-job requests straight to this URL, bypassing the Next.js proxy, to avoid Vercel's ~4.5MB serverless function body limit. Also used as a fallback if `BACKEND_BASE_URL` is unset. Must be a publicly reachable backend URL in production |

`.env.example` / `.env.local.example` in each package show the expected shape — copy and fill in
real values rather than committing the real files.

## Project layout

```
frontend/app/            Next.js App Router pages and API route proxies
frontend/components/     UI components (audio player/recorder, chord diagrams, etc.)
frontend/hooks/          Custom React hooks (e.g. live pitch analysis)
backend/app/main.py      FastAPI entry point — CORS, routers
backend/app/inference.py Core pipeline: audio → analysis → generation → MIDI/WAV
backend/app/model/       Model definitions (active + legacy stacks)
backend/tests/           pytest suite
```

See `CLAUDE.md` for the full architecture reference (endpoint list, checkpoint schemas, token
vocabulary, request flow) used to brief AI coding assistants working in this repo.

## Known limitations

- Model checkpoints aren't included — you need to train your own via the notebook or supply
  compatible `cvae_weights.pth` / `iddm_ppo_weights.pth` files.
- This is a single-tenant app with no authentication. Transcription job metadata/queue/input-audio
  storage are swappable (SQLite/Postgres, in-process/Redis, local disk/S3-compatible — see
  `CLAUDE.md`'s "Async Transcription Job Workflow"); the worker also mirrors its generated MIDI/WAV
  output into the same object storage (when configured) so `/api/download` still works when the web
  service and worker are separate Render services. Everything else — uploads via `/api/upload`, and
  MIDI/WAV generated by the still-synchronous `/generate-variants` path — is still stored directly
  on local disk under `backend/data/`, unchanged from before. That's fine for the current
  single-web-instance deployment but would need the same storage-abstraction treatment to survive
  the web service itself running as multiple instances or being replaced across deploys.
- Test coverage is still growing — see project history for what's actively being hardened.

## Deployment

- **Frontend**: Vercel (`frontend/vercel.json`)
- **Backend web service**: Docker (`backend/Dockerfile`), Python 3.10-slim, exposes port 8000. Set
  `RUN_WORKER_IN_PROCESS=false` in production.
- **Backend worker service**: a second Render Background Worker service on the same image, start
  command overridden to `python -m app.worker_main`. See `CLAUDE.md`'s Deployment section for the
  full picture (Postgres for job metadata, Redis for the job queue, S3/R2 for job input audio).
