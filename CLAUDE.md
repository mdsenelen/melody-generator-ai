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

`_read_audio_bytes` (the shared decoder used by every audio-processing entry point) truncates decoded audio to `MAX_ANALYSIS_DURATION_SEC` (default 30s, env-overridable) before any analysis runs, and probes the source file's header (via `soundfile`, no decode) to pass that cap straight into `librosa.load` so decode work itself is bounded rather than decoding the full upload before slicing. Basic Pitch inference runs at roughly realtime speed on Render's free-tier CPU, so an uncapped clip can exceed `GENERATION_TIMEOUT_SECONDS` and Render's own ~150s gateway timeout — the cap keeps worst-case processing time bounded regardless of upload length. This was previously set equal to `GENERATION_TIMEOUT_SECONDS` (60s each), leaving no headroom for decode/analysis overhead on top of a "roughly realtime" ~60s of Basic Pitch inference; full-length uploads reliably hit the 60s timeout in production, which is why this is now half of `GENERATION_TIMEOUT_SECONDS`. `_transcribe_and_mood`'s result (and the `/api/transcribe` response) includes `source_duration_sec` (the full uploaded clip's duration) and `truncated: bool` alongside `duration_sec` (the duration actually analyzed), so the frontend can tell the user when their clip was cut short.

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

### Frontend (`frontend/app/`)

Next.js App Router. Most backend communication is proxied through Next.js API routes in `frontend/app/api/` via `_lib/backend.ts`. The exceptions are the two calls that carry the full audio file — upload and transcribe — which the browser sends directly to the FastAPI backend (see below), because Vercel's serverless functions have a hard, non-configurable ~4.5MB request body limit that large audio files can exceed. Any other route that ends up carrying a full file body (not just an `id`/`filename` reference) needs the same treatment, or it will hit the same limit.

| Route | Purpose |
| --- | --- |
| `/` (`page.tsx`) | Upload/record audio, display transcription results |
| `/choose-progression` | Select a chord progression |
| `/generate-variants` | Generate melody variants |
| `/listen-progressions` | Playback and analysis |

**Direct-to-backend calls** (bypass the Next.js proxy, see above), both built via `frontend/app/lib/backendUrl.ts`'s `getPublicBackendApiUrl`:

- `frontend/app/lib/upload.ts` posts the audio file straight to `POST {NEXT_PUBLIC_BACKEND_URL}/api/upload`, returning `{id, filename}`.
- `frontend/app/page.tsx`'s `transcribeFile` posts the audio file straight to `POST {NEXT_PUBLIC_BACKEND_URL}/api/transcribe`, returning the analysis result.

There are no `frontend/app/api/upload/route.ts` or `frontend/app/api/transcribe/route.ts` — both were removed when these fixes were made.

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

Upload flow: browser → `POST {NEXT_PUBLIC_BACKEND_URL}/api/upload` directly (not proxied) → stores `upload_{uuid}{ext}` → returns `{id, filename}` → `id` is passed through the normal Next.js proxy to `/api/generate` to reference the stored file. The homepage also immediately transcribes the same file via `POST {NEXT_PUBLIC_BACKEND_URL}/api/transcribe` directly (not proxied, for the same file-size reason) to show analysis results.

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
- **Max analyzed audio duration:** `MAX_ANALYSIS_DURATION_SEC` env var, defaults to 30s — clips longer than this are truncated before analysis; see `_read_audio_bytes` in Architecture above. Deliberately half of `GENERATION_TIMEOUT_SECONDS`, to leave headroom for decode/analysis overhead on top of Basic Pitch's own roughly-realtime inference time.
- **Generation timeout:** `GENERATION_TIMEOUT_SECONDS` env var, defaults to 60s. Measured in production, Render's own platform-level request timeout sits close to 100s, and CPU starvation on the free tier can add ~20-30s of lag before our own timeout error is actually delivered — so this needs real margin below Render's limit, not just to be lower than it, or Render's raw CORS-header-less 502 wins the race instead of our clean JSON error. The frontend retries a failed `/transcribe` call once (same for the initial `/api/upload` call), since a cold Render instance can fail the very first request outright while it's still waking up, and a retry right after is effectively always fast — except when the failure was our own 504: that means `GENERATION_TIMEOUT_SECONDS` already fired after the backend spent a full timeout budget on this exact file, and the underlying call keeps running server-side regardless (the timeout can't cancel it, see `_run_generation`), so an immediate retry would only queue identical work behind it instead of recovering. `/transcribe` does not retry on a 504 for this reason.

---

## Deployment

- **Frontend:** Vercel (`frontend/vercel.json`)
- **Backend:** Docker (`backend/Dockerfile`) — Python 3.10-slim, exposes port 8000
- **Frontend Docker:** `frontend/Dockerfile` — node:18-alpine, runs `npm run dev`
