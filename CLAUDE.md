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

Next.js App Router. Most backend communication is proxied through Next.js API routes in `frontend/app/api/` via `_lib/backend.ts`. The one exception is audio upload: the browser calls the FastAPI backend directly for `POST /api/upload` (see below) because Vercel's serverless functions have a hard, non-configurable ~4.5MB request body limit that large audio files can exceed.

| Route | Purpose |
| --- | --- |
| `/` (`page.tsx`) | Upload/record audio, display transcription results |
| `/choose-progression` | Select a chord progression |
| `/generate-variants` | Generate melody variants |
| `/listen-progressions` | Playback and analysis |

**Direct-to-backend upload** (bypasses the Next.js proxy, see above): `frontend/app/lib/upload.ts` posts the audio file straight to `POST {NEXT_PUBLIC_BACKEND_URL}/api/upload` (URL built by `frontend/app/lib/backendUrl.ts`), returning `{id, filename}`. There is no `frontend/app/api/upload/route.ts` — it was removed when this fix was made.

**API route proxies** (each forwards to FastAPI):

- `POST /api/transcribe` — analyze uploaded audio
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

Upload flow: browser → `POST {NEXT_PUBLIC_BACKEND_URL}/api/upload` directly (not proxied) → stores `upload_{uuid}{ext}` → returns `{id, filename}` → `id` is passed through the normal Next.js proxy to `/api/generate` to reference the stored file.

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

---

## Deployment

- **Frontend:** Vercel (`frontend/vercel.json`)
- **Backend:** Docker (`backend/Dockerfile`) — Python 3.10-slim, exposes port 8000
- **Frontend Docker:** `frontend/Dockerfile` — node:18-alpine, runs `npm run dev`
