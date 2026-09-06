# Plan: split full-audio transcription from short-clip analysis

Status: **APPROVED 2026-09-06.** Sub-decisions resolved (below). Step 1 in progress.

## Why

Today every audio path (transcription, mood/key/BPM/chords, variant generation)
runs on one decoded buffer that `_read_audio_bytes` truncates to
`MAX_ANALYSIS_DURATION_SEC`. Consequences:

- The user's uploaded audio is never transcribed in full — the MIDI stops at
  the cap.
- Peak memory scales with clip length, which is what pushed the free-tier
  512 MiB instance into repeated `oomKilled` restarts (see the "Prod OOM
  incident" row in `PROGRESS.md`).

We want:

1. **Full audio → full MIDI**, uncropped, at bounded (length-independent)
   memory — via chunked Basic Pitch inference.
2. **Short clip → analysis + generation** (mood, key, BPM, chords, variants) on
   a user-chosen excerpt, re-runnable without re-transcribing.

## Decisions (confirmed)

| # | Decision |
|---|---|
| 1 | **Separate endpoints.** Transcription is one async job. Clip analysis is its own endpoint, callable repeatedly against different clips of the same upload without re-transcribing. |
| 2 | **Hard upload-duration cap: 10 minutes.** `MAX_UPLOAD_DURATION_SEC` env var, default `600`. Uploads longer than this are rejected at `/api/upload` (or at job creation) with a clear 400. |
| 3 | **Staged rollout, 7 steps, feature-flagged** (`TRANSCRIBE_CHUNKED`), chunked path off by default until proven in prod. |
| 4 | This doc is written and approved before step 1. |

---

## Architecture

### Endpoints

| Endpoint | Shape | Work | Memory |
|---|---|---|---|
| `POST /api/transcribe` | async job → `{job_id}`; poll `GET /api/transcribe/{job_id}` | **Full audio**, chunked Basic Pitch → full-length MIDI. No analysis fields. | flat ~300–400 MB regardless of length |
| `POST /api/analyze` | **synchronous** → analysis JSON directly | mood, key, BPM, chords, pitch histogram on `[clip_start_sec, clip_end_sec]` of an existing upload | ~librosa-features only (see below) |
| `POST /api/generate-variants` | unchanged (still async-job-wrapped for the result page) | now also takes `clip_start_sec`/`clip_end_sec`; runs its internal Basic Pitch on the clip only | serialized by `HEAVY_WORK_LOCK` |

`/api/transcribe`'s result loses `mood_label`, `mood_idx`, `key`, `tempo_bpm`,
`detected_chords`, `pitch_histogram`, `average_pitch` — those move to
`/api/analyze`. It keeps / gains: `midi_b64` (full length), `midi_filename`,
`wav_b64?`, `source_duration_sec`, `n_notes`, `n_chunks`, `truncated` (now
always `false` unless the 10-min cap hit).

### `/api/analyze` — Basic-Pitch-free (decided)

**Decision: librosa-only, no Basic Pitch in the analysis path** — putting
`tflite` inference back into a re-runnable endpoint would reinstate the memory
problem. The clip-analysis outputs don't need note-level transcription:

| Output | Current source | Clip-analysis source (proposed) |
|---|---|---|
| `detected_chords` | `_detect_chords_from_audio` (librosa chroma) | **unchanged** — already Basic-Pitch-free |
| `tempo_bpm` | `_estimate_tempo(note_events)` | `librosa.beat.beat_track` on the clip |
| `key` | music21 on the MIDI, or Krumhansl-Schmuckler on the pitch histogram | Krumhansl-Schmuckler on librosa chroma of the clip |
| `pitch_histogram` | from note events | from librosa chroma / CQT of the clip |
| `average_pitch` | mean of note-event pitches | spectral-centroid-derived estimate, or chroma-weighted |
| `mood_label` | `heuristic_mood_from_metrics(tempo, avg_pitch, key)` | same, fed by the above |

This makes `/api/analyze` genuinely light — librosa chroma + beat tracking on a
30–60 s clip is ~2–5 s and adds maybe 50–80 MB over the fixed library floor, no
`tflite` inference. Served **synchronously**, and light enough that it does
**not** take `HEAVY_WORK_LOCK` (no `tflite`, no torch). Accept the small
accuracy trade on `tempo`/`key` vs. Basic-Pitch-derived values — a
Basic-Pitch-backed mode is explicitly *not* being added (it would reinstate the
memory problem it's meant to avoid).

---

## Chunked transcription — the core

### The problem, precisely

Basic Pitch's `predict(path, model)` on an isolated slice `[t_a, t_b]`:

1. streams the slice in ~2 s internal windows (input side is already
   memory-frugal),
2. **accumulates the full-slice posteriorgram** — `onset` and `note` arrays of
   shape `[frames, 88]`, `contour` of `[frames, 264]`, at ~86 frames/s. That is
   ~18 MB per minute of audio for the three arrays, and `model_output_to_notes`
   roughly doubles it transiently. **This is the length-proportional memory
   term.** A 10-min file's posteriorgram alone is ~180 MB, plus note extraction
   → OOM on 512 MiB.
3. `model_output_to_notes` peak-picks onsets, tracks each note forward through
   the `note` posteriorgram until it decays, applies `minimum_note_length`
   (~128 ms), returns `note_events` = list of `(start_s, end_s, pitch_midi,
   amplitude, pitch_bends)`.

Running `predict()` on chunks keeps that posteriorgram bounded to one chunk's
worth (~9 MB for 30 s), discarded per chunk — **only the merged note list
accumulates, and note tuples are tiny** (<1 MB for a 10-min song).

### Boundary artifacts (why naive concatenation fails)

Splitting audio at hard cuts and concatenating `note_events` produces three
distinct errors:

- **A — false offset.** A note still sounding at `t_b` is reported by chunk *k*
  with `end ≈ t_b` because the posteriorgram simply ends there. The note is
  truncated.
- **B — false onset.** A note that started *before* `t_a` (carried in from the
  previous chunk) is reported by chunk *k+1* with `start ≈ t_a`: the chunk's
  audio "begins" with the note at full energy, and either the edge produces a
  spurious onset spike or the melodia backward-pass assigns an onset at the
  first frame. A continuation is misread as a new note.
- **C — edge degradation.** Pitch/timing accuracy is worse in the first/last
  ~0.5 s of anything Basic Pitch processes (CNN receptive field / padding).

### Solution: overlapping windows + a left-to-right weld/dedup merge

This is the standard approach for chunked sequence models (chunked ASR,
diarization, source separation face the identical problem). Two parts:
**overlap** so every true boundary has clean audio on both sides, and a
**merge pass** that reconciles A and B.

Rejected alternative — *stitch the raw posteriorgrams and call
`model_output_to_notes` once*: musically ideal (note tracking stays genuinely
continuous), but it rebuilds the full-length posteriorgram in memory, which
defeats the entire point, and it depends on `run_inference` / `unwrap_output`
internals that are not stable public API. Not worth the fragility.

#### Chunk geometry

```
CHUNK_SEC   = 30.0     # audio fed to each predict() call
OVERLAP_SEC = 4.0      # consecutive chunks share this much audio
HOP_SEC     = CHUNK_SEC - OVERLAP_SEC          # = 26.0
```

Chunk *k* covers `[k·HOP, k·HOP + CHUNK_SEC]`, clamped to `[0, duration]`.
Decoded per chunk with `librosa.load(buf, sr, offset=k·HOP, duration=CHUNK_SEC)`
— only that window is ever decoded.

- 10-min (600 s) file → `ceil((600 − 30) / 26) + 1 = 23` chunks.
- Recompute overhead: `OVERLAP_SEC / HOP_SEC ≈ 15 %` extra Basic Pitch compute.
- **Handoff point** between chunk *k* and *k+1*:
  `B_k = s_{k+1} + OVERLAP_SEC / 2` (middle of the shared region). Chunk *k*
  "owns" note starts before `B_k`; chunk *k+1* owns starts at/after `B_k`.

`OVERLAP_SEC = 4` comfortably exceeds the ~0.5 s edge-degradation zone on each
side and gives the weld logic room. It does **not** need to exceed the length
of the longest sustained note — the weld chains truncated fragments across any
number of chunks (see below), so a 30 s drone that spans three chunks still
comes out as one note.

#### The merge algorithm

Process chunks left to right. Keep `merged: list[Note]` (absolute times).
For chunk *k ≥ 1*, let `notes_k` be its `predict()` output shifted by `+s_k`:

**1. Classify `notes_k`:**
- `left_edge` = notes with `start ≤ s_k + EDGE_IN` — candidates for welding
  (born at the chunk's left edge; likely continuations)
- `body` = the rest

**2. Weld pass** — for each `n` in `left_edge`, look in `merged` for `m` where:
- `m.pitch == n.pitch`, and
- `m.end ≥ e_{k-1} − EDGE_ε` — `m` was **truncated by the previous chunk's
  right edge** (this is the key disambiguator: if chunk *k−1* ended `m`
  cleanly, well before its own edge, then a same-pitch note at chunk *k*'s
  start is a genuine re-articulation, not a continuation), and
- `n.start ≤ m.end + GAP` — they overlap in time or are near-contiguous
  (with a 4 s overlap they overlap heavily; `GAP` covers the small-gap case
  where Basic Pitch dropped a frame or two at the seam), and
- `n.end > m.end` — `n` genuinely extends the note (if `n.end ≤ m.end`, `n`
  is a fully-contained duplicate → drop it).

  → **weld:** `m.end = n.end`; `m.amplitude = duration-weighted mean`;
  `m.pitch_bends = m.bends[:orig_end] ++ n.bends[from seam]`; consume `n`.

  `left_edge` notes with **no** matching `m` → real notes that happen to start
  near the edge → move them to `body`.

**3. Dedup + commit `body`** — for each `n` in `body`:
- if `merged` already has `m` with `m.pitch == n.pitch`,
  `|m.start − n.start| ≤ DEDUP` and `|m.end − n.end| ≤ DEDUP` → same note,
  already captured from chunk *k−1*'s overlap → skip `n` (optionally average
  amplitude/timing),
- else append `n` to `merged`.

**4. (optional, gated on testing) overshoot trim** — a `merged` note `m` from
chunk *k−1* with `m.end > B_{k−1}` that was **not** welded: its offset landed
in territory we now trust chunk *k* for. If chunk *k* has no continuation →
clamp `m.end = B_{k−1}`. Ship without this; add only if test diffs show
tail-hallucination is real.

**Chaining across ≥3 chunks:** a very long note is truncated at chunk *k−1*'s
edge (in `merged`, `end ≈ e_{k−1}`), appears in chunk *k*'s `left_edge`, welds
to `e_k` (also truncated), appears in chunk *k+1*'s `left_edge`, welds to
`e_{k+1}`, … terminates naturally in whichever chunk the note actually ends.

**Constants (starting values — tuned empirically in step 2):**

```
EDGE_IN = 0.30 s   # "born at the left edge" window
EDGE_ε  = 0.15 s   # "truncated at the right edge" tolerance
GAP     = 0.15 s   # max seam gap to treat two fragments as one (≈ min note len)
DEDUP   = 0.15 s   # same-note tolerance for full-overlap duplicates
```

#### Assembling the output MIDI

One `pretty_midi.PrettyMIDI` (via `_tokens_to_midi_bytes`), one non-drum
`Instrument`, all `merged` notes (absolute seconds). Tempo is cosmetic (note
times are in seconds) — **use the default clip's tempo** from the clip analysis
(decided; no extra whole-file `librosa.beat` pass). `n_notes` and `n_chunks` go
in the job result.

#### Residual risks (honest)

- **Fast repeated notes landing on a boundary** (tremolo, trill). `EDGE_IN` is
  deliberately small (0.30 s) and the weld requires `m` to be edge-truncated,
  so the exposure is a ~0.3 s window at each fixed chunk boundary. Worst case:
  1–2 wrong notes out of thousands, at a predictable timestamp. Acceptable.
- **A real onset within a few ms of `B_k`.** Both chunks see it with full
  context (that's what the overlap buys); dedup catches the double. True loss
  only if *both* chunks sub-threshold it — which also happens without chunking.
- **Constants are empirical.** They need a real multi-minute track to tune.
- **Determinism:** chunk boundaries are a pure function of file length, so a
  given file always chunks identically → reproducible output.

#### Validation (part of step 2, gates the flag flip in step 5)

1. 3–4 reference clips, 2–5 min: sparse/monophonic, dense/polyphonic, long
   pads, fast passages.
2. Transcribe each **whole** (cap lifted, on a 2 GB box or locally) = ground
   truth, and **chunked** on the 512 MiB target.
3. Score with `mir_eval.transcription` (already a Basic Pitch dependency):
   note precision/recall/F1 at ±50 ms onset + exact pitch. Separately count
   "notes differing only within `OVERLAP_SEC` of a boundary."
4. Gate: chunked-vs-whole **F1 ≥ 0.95** and boundary-specific diffs **< 1 %**
   of notes. Tune constants to hit it.
5. Audible A/B at each boundary timestamp on the rendered WAV.

---

## Memory — does it fit 512 MiB?

```
fixed library floor   ~250–350 MB   numba×2 (librosa + resampy), scipy,
                                    scikit-learn, tflite-runtime, music21,
                                    pretty_midi — paid once per process,
                                    NOT reduced by chunking
one 30 s chunk         ~30–60 MB    decode + posteriorgram + note extraction
merged note list       < 1 MB       even for a 10-min song
------------------------------------------------------------------
transcription peak     ~300–420 MB  FLAT in input length
```

vs. today's unbounded growth (~+18 MB/min posteriorgram + decode + frames).

- **Transcription becomes structurally safe on 512 MiB** — the specific failure
  mode (length-proportional OOM) is designed out. This is the real win.
- Generation (torch, ~450–480 MB peak) stays fragile but is serialized by
  `HEAVY_WORK_LOCK`, lower priority, and currently 503s on missing weights.
- **`malloc_trim(0)` between chunks** (reuse `_release_memory_to_os`) to stop
  glibc holding each chunk's freed arena.
- **Step 1 measures the real fixed floor** (one RSS probe after warm-up, in
  prod). If it's ~400 MB rather than ~280 MB, margin gets thin and Render
  Standard re-enters the conversation. Everything downstream is contingent on
  that number.

---

## Job store changes (blocker — must land before chunked transcription)

`JobStore` today has **no lease renewal and no incremental progress**.
`claim_job` hard-sets `progress = 50`; `mark_completed` sets `100`.
`DEFAULT_LEASE_SECONDS = 900`. A 10-min file at ~realtime Basic Pitch + 15 %
overlap ≈ **11–12 min of compute > the 900 s lease** → the lease expires
mid-job, another worker reclaims it, and it **re-runs from scratch**, possibly
forever.

Add one method to the `JobStore` protocol + `SQLJobStore` (one class, both
SQLite and Postgres via `self._q()`; tests use `create_sqlite_job_store`):

```python
def heartbeat(self, job_id: str, *, lease_token: str,
              progress: int, lease_seconds: float) -> bool
```

`UPDATE transcription_jobs SET lease_expires_at = ?, progress = ?, updated_at = ?
 WHERE id = ? AND lease_token = ? AND status = 'processing'` — returns `False`
if the lease was already reclaimed. (Step 1: worker just logs a warning and
carries on — the eventual `mark_completed` lease check is the real safety net,
and with a 180 s/chunk lease a mid-job reclaim is essentially impossible. A
clean early-abort path can harden this later.)

The worker calls `heartbeat` after each chunk:
`progress = int(90 * chunks_done / n_chunks)` (last 10 % = MIDI assembly +
object-storage upload), `lease_seconds = 180` (generous per-chunk margin; a
genuinely dead worker is now reclaimed in ~1 chunk-time, not 15 min).

Tests: heartbeat extends the lease; heartbeat with a stale token returns
`False` and does not touch the row; a job that heartbeats past the original
lease still completes and isn't double-claimed.

---

## Frontend

`/analyse` goes from "one result" to **two independent results + a clip
selector**:

1. Upload → auto-start `/api/transcribe` (full MIDI, background). Progress bar
   driven by real `progress` (`n/n_chunks`).
2. Clip selector, default `0 → min(60, duration)`: **dual-thumb range control**
   (`components/clip-range.tsx`) over a plain duration bar. No dependency.
   Controlled, emits `{startSec, endSec}`. **This is the permanent UI**
   (decided) — a canvas waveform is deferred to roadmap Phase 7 (`/audio-viz`)
   rather than built half-now and rewritten later.
3. "Analyze clip" → `POST /api/analyze` (sync, ~2–5 s) → renders mood / key /
   BPM / chords for that clip. Re-runnable: move the handles, click again.
4. "Generate variants" → `/api/generate-variants` with the same clip bounds.
5. When the transcribe job completes → "Download full MIDI" + WAV preview.

`hooks/use-job-result.ts` and GP3's `/result/[jobId]` page: the transcription
result shape shrinks (MIDI/WAV only, no analysis fields). Update
`app/lib/jobResult.ts`'s `TranscriptionResult` type, `result-view.tsx`'s
transcription branch, and the RTL tests. The clip-analysis result is not
job-based (synchronous) — it renders inline on `/analyse`, no `/result` page.

Every async op still cancellable (`AbortController`), explicit
idle/loading/error/empty states (non-negotiable #3), keyboard + focus + ARIA
live region on the clip selector (non-negotiable #4).

---

## Contract changes

Backward-compatible **in two ordered steps** (frontend and backend deploy
separately — release gate):

1. **Backend first:** add `/api/analyze`; add optional `clip_start_sec` /
   `clip_end_sec` to `/api/generate-variants` and `/api/generate-progression`
   (default to `0`..`min(60, dur)` when absent — old frontend keeps working);
   `/api/transcribe` result **still includes** the analysis fields for now
   (computed on the default clip) so the old frontend's result page doesn't
   break.
2. **Frontend switches** to calling `/api/analyze` separately and rendering the
   slimmer transcription result.
3. **Backend cleanup:** drop the analysis fields from `/api/transcribe`'s
   result once no client reads them.

Shared TS types (`frontend/app/lib/*.ts`) updated in the same commit as each
backend contract change (scope rule).

`MAX_UPLOAD_DURATION_SEC` (default `600`) enforced at `/api/upload`
(probe duration from the header, 400 if over) — reject early, before a job
exists.

---

## Staged plan (5 steps, each its own commit, TDD)

| # | Step | Gate |
|---|---|---|
| **1** | **Chunked transcription behind `TRANSCRIBE_CHUNKED=false` + `JobStore.heartbeat` + measure real prod peak.** `_transcribe_and_mood_chunked` (chunk-window decode via `librosa.load(offset=, duration=)`, per-chunk `_run_basic_pitch_predict`, `_merge_chunk_notes` weld/dedup, `_tokens_to_midi_bytes` assembly, default-clip analysis reusing the merged notes + one `librosa.load(duration=60)` for chords, no WAV render). `heartbeat` on the store (protocol + `SQLJobStore` SQLite **and** Postgres), called per chunk from the worker. Unit tests: `_merge_chunk_notes` (pure fn, synthetic boundary-crossing inputs) + `heartbeat` (extend / stale-token / past-original-lease). Deploy flag-on, transcribe a synthetic 5-min clip in prod, **report the measured peak RSS**. | merge + heartbeat unit tests green; synthetic 5-min clip → full-length MIDI in prod; **peak RSS number reported** → go / revisit Standard |
| 2 | **`/api/analyze` (librosa-only, sync) + `clip_*` params on generate endpoints + `MAX_UPLOAD_DURATION_SEC` at `/api/upload`.** librosa paths for `tempo`/`key`/`pitch_histogram`/`avg_pitch`. Contract step 1 (additive — `/api/transcribe` result unchanged for now). Shared TS types. | `/api/analyze` returns sane mood/key/BPM/chords on fixture clips; `/api/transcribe` result byte-identical; 400 on a >10-min upload |
| 3 | **Frontend: `clip-range.tsx` + two-result `/analyse` + slim transcription result.** wire `/api/analyze` (re-runnable), update `jobResult.ts` + GP3 `result-view.tsx` + RTL. Contract step 2. | both results render independently; re-analyze a different clip works; `npm run typecheck` + `npm test` green; keyboard + focus + ARIA-live on the range control |
| 4 | **Flip `TRANSCRIBE_CHUNKED=true` in prod, remove the flag + the old truncating path.** | full-length MIDI on a real 5-min upload in prod; flat memory in Render metrics; no OOM over a day |
| 5 | **Backend cleanup:** drop analysis fields from `/api/transcribe`'s result (contract step 3); update types + tests. | no client reads the dropped fields; types + tests green |

Canvas waveform selector is **out of scope here — deferred to roadmap Phase 7
(`/audio-viz`)** rather than built half-now and rewritten. Append each step to
`PROGRESS.md`.

---

## Sub-decisions — resolved 2026-09-06

1. `/api/analyze`: **librosa-only**, no Basic-Pitch path (would reinstate the
   memory problem).
2. `/api/analyze`: **synchronous** (librosa-only makes it safe), no `HEAVY_WORK_LOCK`.
3. Full-MIDI tempo: **reuse the default clip's tempo**. No whole-file
   `librosa.beat` pass.
4. Clip selector: **range control is the permanent UI**. Waveform → roadmap
   Phase 7, not now.

Full mir_eval reference-clip validation (F1 ≥ 0.95 chunked-vs-whole on real
music) still applies but runs as a follow-up harness — step 1 gates on the
merge unit tests + a synthetic-clip prod run + the memory number.
