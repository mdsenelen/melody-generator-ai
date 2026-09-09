# Plan: split full-audio transcription from short-clip analysis

Status: **All 5 steps done (2026-09-06 → 2026-09-09).** Chunked transcription
peaks at ~400–430 MB on a 5-min clip, flat across chunks — fits the free tier's
512 MB (the project **stays on free tier**, concurrent-request OOM risk closed
in code; see `PROGRESS.md` "Free-tier hardening"). `/api/transcribe` returns
MIDI + `note_events`; `POST /api/analyze` slices those to a clip window and
computes mood/key/tempo/chords with pure arithmetic (no librosa, no Basic
Pitch); `/analyse` shows the two results independently with a re-runnable clip
range. The `TRANSCRIBE_CHUNKED` flag and the old single-pass truncating path
are gone. Follow-ups still open: `mir_eval` chunked-vs-whole F1 validation
harness, `_merge_chunk_notes` `edge_eps` tuning, WAV preview for the analyse
clip, canvas waveform selector (roadmap phase 7).

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
| `POST /api/transcribe` | async job → `{job_id}`; poll `GET /api/transcribe/{job_id}` | **Full audio**, chunked Basic Pitch → full-length MIDI + persisted `note_events` | flat ~300–400 MB regardless of length |
| `POST /api/analyze` | **synchronous** → analysis JSON directly | mood, key, BPM, chords, pitch histogram on `[clip_start_sec, clip_end_sec]` of a completed transcribe job | **arithmetic on stored note events — zero librosa, zero Basic Pitch** (see below) |
| `POST /api/generate-variants` | unchanged (still async-job-wrapped for the result page) | now also takes `clip_start_sec`/`clip_end_sec`; decodes + runs its internal Basic Pitch on that window only | serialized by `HEAVY_WORK_LOCK` |

**Contract, staged:**
- **Step 2 (this step) — additive.** `/api/transcribe`'s result *gains*
  `note_events` (trimmed `{start,end,pitch,velocity}`, ~40 KB for a 5-min
  track); everything it returns today stays. `/api/analyze` is new. Nothing
  breaks.
- **Step 5 — the removal.** `/api/transcribe`'s result then *loses* the
  computed fields (`mood_label`, `mood_idx`, `key`, `tempo_bpm`,
  `detected_chords`, `pitch_histogram`, `average_pitch`) — the client reads
  them from `/api/analyze` by then. `note_events` stays (analyze needs it).

### `/api/analyze` — MIDI-based, no audio at all (revised 2026-09-07)

**Decision: analysis runs on the transcribe job's stored note events, sliced to
`[clip_start_sec, clip_end_sec]` — pure arithmetic, no librosa call, no Basic
Pitch, no audio decode.** This replaces the earlier "librosa-only" design after
a code audit (below) showed the analysis path is *already* ~95 % MIDI-derived —
only chord detection touched raw audio.

Audit of what each analysis output actually uses today (`inference.py`):

| Output | Today's source | Touches raw audio / librosa? |
|---|---|---|
| `tempo_bpm` | `_estimate_tempo` → `pretty_midi.PrettyMIDI(...).estimate_tempo()` on the MIDI, inter-onset-interval median fallback on note events | **No.** `librosa.beat` is used *nowhere* in the codebase |
| `average_pitch` | `np.mean` of note-event pitches | **No** |
| `pitch_histogram` | pitch-class counts from note events | **No** |
| `key` | `_key_from_histogram` — Krumhansl-Schmuckler profile match on the 12-bin histogram (music21 removed in Phase B) | **No** |
| `mood_idx` / `mood_label` | `heuristic_mood_from_metrics(tempo_bpm, avg_pitch, key)` — a 3-branch rule (happy / sad / neutral). No timbre, no audio, no model | **No — already 100 % MIDI-derived** |
| `detected_chords` | `_detect_chords_from_audio` → `librosa.feature.chroma_cqt` on the clip audio | **Yes — the only one** |

So `/api/analyze` needs: the job's persisted `note_events` (small — ~800 notes
for a 5-min track, ~50 KB JSON), a `[clip_start, clip_end]` slice, and the same
arithmetic helpers run on that slice. **New:** `_chords_from_note_events(notes,
window_sec≈2.0)` replaces `_detect_chords_from_audio` — window the notes, take
each window's active pitch classes weighted by sounding duration, template-match
with the existing `_infer_chord_from_chroma` logic (rename → takes a 12-vector).
Reference implementation: the thesis notebook's `extract_chords()` (Cell 13).

Memory / cost: **~0 MB, sub-millisecond, re-runnable for free.** It does *not*
take `HEAVY_WORK_LOCK` and never imports anything new.

Tradeoffs, stated honestly:
- **Chords lose chroma's ear.** `chroma_cqt` hears the actual harmonic spectrum
  (overtones, mix content Basic Pitch dropped); MIDI chords only see the
  transcription. For monophonic / sparse input they're close; for dense
  polyphony chroma was better. Upside: MIDI chords are *consistent* with the
  notes the user sees, and it's exactly what the thesis pipeline did.
- **`librosa` is NOT removed from the process.** `basic_pitch.inference` imports
  it eagerly, `_read_audio_bytes` / `_decode_audio_window` need `librosa.load`
  for mp3/m4a/ogg decode, and there's a `librosa.pyin` fallback. The idle floor
  is unchanged. What this buys: the chunked path's second `clip_audio` decode
  and the `chroma_cqt` transient (~30–60 MB during the analysis phase, to be
  measured during implementation) both go away, and `/api/analyze` becomes a
  genuinely free re-runnable call instead of a 50–80 MB librosa-feature pass.

**Not doing: an ML mood classifier.** There is no mood-labeled dataset — not in
the repo, not anywhere. The notebook (Cells 6, 13) labels CVAE training data
with `tokens_to_mood_idx()` (avg pitch + note count) and
`heuristic_mood_label_from_pm()` (tempo + avg pitch) — both rule-based
heuristics. Training a RandomForest/GBM on those labels just learns to imitate
the heuristic (bounded above by it) and re-adds `scikit-learn` (~40–50 MB
import). A genuinely better classifier needs an external emotion corpus
(EMOPIA / DEAM / VGMIDI) with a different label taxonomy (valence–arousal
quadrants ≠ happy/sad/neutral) — a separate project, out of scope here.
Optional cheap win instead: enrich `heuristic_mood_from_metrics` with note
density, mean absolute interval, pitch range, and rhythmic regularity (all
note-event arithmetic) — still no model, still no librosa.

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

## Memory — measured on free tier (512 MiB)

> **Resolved, step 1 (2026-09-06):** a chunked 5-minute transcription peaked at
> **~400–430 MB, flat across all 12 chunks** — memory does not scale with
> length, which was the point. ~80–85% of the 512 MiB cap; ~100 MB headroom —
> enough for the one-at-a-time transcription path, not for anything concurrent.
>
> **Decision: stay on free tier.** The concurrent-request risk is closed in
> code instead (see `PROGRESS.md` "Free-tier hardening"): `_run_generation`
> 429s immediately when it can't get `HEAVY_WORK_LOCK`; `run_basic_pitch`
> waits `WORKER_HEAVY_WORK_WAIT_SEC` then re-queues; generation 503s *before*
> importing torch when weights are absent. `malloc_trim` runs between chunks
> and after every heavy task.
>
> **Phase B (done, 2026-09-07):** import-surface audit against a fresh
> interpreter (`/proc/self/status` VmRSS). Idle floor is ~290 MB after
> warm-up (numba, llvmlite, scipy, tflite_runtime, mir_eval, librosa,
> pretty_midi, resampy + the warm-up's resident Basic Pitch model). **`torch`
> is never imported on this path** (lazy-torch fix confirmed by measurement),
> and `matplotlib` / `scikit-learn` / `tensorflow` are never imported at all.
> The only removable weight was **`music21` (−35 MB RSS at boot)** — dropped;
> `_key_from_histogram` (Krumhansl-Schmuckler) is now the sole key detector.
> Everything else on the list is pulled eagerly by `basic_pitch.inference` and
> can't be deferred without moving the cost into the transcription peak.
> **`TRANSCRIBE_CHUNK_SEC` 30→15 measured: no peak benefit** (per-chunk
> working set ~11 MB either way) — kept at 30. No per-transcription ratchet
> (4 sequential runs flat). `torch` in the image (~200 MB disk) is dead
> weight but removing it is a "generation permanently off" call, left to the
> user.

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
| ~~1~~ | ✅ **DONE 2026-09-06** (PR #14 + prod flag flip). `_merge_chunk_notes` weld/dedup + `_transcribe_full_chunked` + `JobStore.heartbeat` + 8 tests. 5-min synthetic clip → 788 notes / 12 chunks / full 300.0s, **peak RSS ~400–430 MB flat**. Decision from the number: **stay free**, harden concurrency in code (429-on-busy + weights-before-torch — see `PROGRESS.md` "Free-tier hardening", landed 2026-09-07). `_merge_chunk_notes` `edge_eps` was ~50 ms too tight on the synthetic clip — tuning it stays with the mir_eval validation harness (needs an F1 measurement on real music to set, not a guess), not step 2. | *(met)* |
| ~~1b~~ | ✅ **Phase B DONE 2026-09-07.** Import audit (fresh interpreter, VmRSS): only `music21` was removable — **−35 MB at boot**, dropped (`_key_from_histogram` is now the sole key detector). `torch` confirmed never imported here; `matplotlib`/`sklearn`/`tensorflow` never imported at all. `TRANSCRIBE_CHUNK_SEC` 30→15 measured — **no peak benefit** (per-chunk working set ~11 MB either way), kept at 30. No per-transcription ratchet. Still open, non-blocking: `torch` out of the Docker image (~200 MB disk, "generation off" decision — user's call); `healthCheckPath=/health` (dashboard-only). | *(met — peak explained, −35 MB idle, no regression)* |
| 2 | **`/api/analyze` (MIDI-based, sync) + `clip_*` params on `/generate-variants` + `MAX_UPLOAD_DURATION_SEC` at `/api/upload`.** `POST /api/analyze {job_id, clip_start_sec?, clip_end_sec?}` → `analyze_clip()` slices the job's stored `note_events` to the window and reruns `_estimate_tempo` / `_key_from_histogram` / `heuristic_mood_from_metrics` / `_pitch_histogram` on the slice — no librosa, no audio, does **not** take `HEAVY_WORK_LOCK`. New `_chords_from_note_events` replaces `_detect_chords_from_audio` in `_transcribe_and_mood` + `_transcribe_and_mood_chunked` (keep the audio version for `generate_from_audio`). Persist `note_events` (trimmed `{start,end,pitch,velocity}`) in the transcribe job result. `MAX_UPLOAD_DURATION_SEC` best-effort at `/api/upload` (header probe — hard enforcement stays in the chunked transcribe path for mp3/m4a the probe can't read). `clip_start_sec`/`clip_end_sec` on `/generate-variants` → decode only that window for generation. Contract step 1 (additive — `/api/transcribe` gains `note_events`, keeps everything else). Shared TS types. | `/api/analyze` returns sane mood/key/BPM/chords off note events with **zero new imports** (subprocess test); slicing to a sub-window changes the numbers; `/api/transcribe` result is a superset of today's; 400 on a >10-min wav upload; `pytest` + `npm run typecheck` green |
| 3 | **Frontend: `clip-range.tsx` + two-result `/analyse` + slim transcription result.** wire `/api/analyze` (re-runnable), update `jobResult.ts` + GP3 `result-view.tsx` + RTL. Contract step 2. | both results render independently; re-analyze a different clip works; `npm run typecheck` + `npm test` green; keyboard + focus + ARIA-live on the range control |
| ~~4~~ | ✅ **DONE 2026-09-09.** `TRANSCRIBE_CHUNKED` const + the `if/else` branch in `run_basic_pitch` removed — `_transcribe_and_mood_chunked` is the only path. `_transcribe_and_mood` stays (webm-recording fallback + generation-path transcription). No WAV preview for the full-length MIDI (`wav_b64: null`, `wav_filename: ""`). Flag was already `true` in prod since step 1, so this is code cleanup, not a behaviour change. README / CLAUDE.md / architecture-reference updated. | *(met — prod already chunked, `n_chunks` in every result; verified via a fresh prod transcribe)* |
| ~~5~~ | ✅ **DONE 2026-09-09.** `run_basic_pitch`'s result dropped `mood_label`, `mood_idx`, `detected_chords`, `key`, `pitch_histogram`, `tempo_bpm`, `average_pitch` — kept `n_notes`, `duration_sec`, `source_duration_sec`, `truncated`, `midi_b64`/`midi_filename`, `wav_b64`/`wav_filename` (null/empty), `n_chunks`, `note_events`. `_transcribe_and_mood_chunked` no longer computes them either. TS `TranscriptionResult`: 7 fields removed, `note_events` now required, `n_chunks?` added. `result-view.tsx` transcription branch: MIDI download only (no WAV, no analysis line). Backward-compatible either deploy order — the frontend stopped reading these in step 3, so an old field present-but-unread or a new field absent-but-not-read both no-op. `typecheck` confirms no non-test code touched them. Backend + frontend tests updated. | *(met — `typecheck` clean, 78 frontend tests, `/api/analyze` still serves everything off `note_events`)* |

Canvas waveform selector is **out of scope here — deferred to roadmap Phase 7
(`/audio-viz`)** rather than built half-now and rewritten. Append each step to
`PROGRESS.md`.

---

## Sub-decisions — resolved 2026-09-06, `/api/analyze` revised 2026-09-07

1. `/api/analyze`: **MIDI-based** — arithmetic on the transcribe job's stored
   `note_events`, sliced to the clip window. No Basic Pitch, no librosa, no
   audio decode. (Was "librosa-only" until the 2026-09-07 code audit showed
   analysis is already ~95 % MIDI-derived — see the `/api/analyze` section
   above.)
2. `/api/analyze`: **synchronous**, no `HEAVY_WORK_LOCK` (it's pure arithmetic).
3. Full-MIDI tempo: **reuse the default clip's tempo**. No whole-file
   `librosa.beat` pass. (`librosa.beat` is used nowhere in the codebase.)
4. Clip selector: **range control is the permanent UI**. Waveform → roadmap
   Phase 7, not now.

Full mir_eval reference-clip validation (F1 ≥ 0.95 chunked-vs-whole on real
music) still applies but runs as a follow-up harness — step 1 gates on the
merge unit tests + a synthetic-clip prod run + the memory number.
