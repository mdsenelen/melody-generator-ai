# ADR 0001: One typed state machine for the audio → melody pipeline

- Status: accepted (implemented 2026-11-10)
- Date: 2026-11-07
- Phase: roadmap Phase 1 (`/domain-model`)

## Context

`/analyse` and `/generate-variants` run the same pipeline: pick a source, transcribe it (in the browser, falling back to the server), then either analyse a clip window or generate variants. Each page tracks that run with its own loose fields:

- `/analyse`: `isTranscribing`, `statusMessage`, `errorMessage`, `transcription`, plus three refs (`transcribeReqRef`, `mountedRef`, `transcribeAbortRef`) and two closure flags (`uploadPhase`, `inBrowser`) that decide which status copy may overwrite which.
- `/generate-variants`: `loading`, `progressLabel`, `error`, `result`, plus a source spread over five fields (`selectedFile`, `selectedSourceName`, `storedFilename`, `storedUploadId`, `useStoredUpload`).

### Impossible states the current code allows

1. **A half-stored source:** `storedUploadId` set while `storedFilename` is `null`, or `useStoredUpload === true` while `selectedFile` is also set. Generation then follows whichever `if` branch it reaches first. Today it's prevented only by every handler remembering to reset all five fields together.
2. **Stale progress from an earlier run:** a superseded transcription can still write `statusMessage` unless every callback checks `requestId === transcribeReqRef.current && mountedRef.current`. There are 9 such checks in `runTranscription`; missing one makes the old run's progress overwrite the new one.
3. **Busy with no description:** `loading === true` with `progressLabel === null` while browser transcription is running, so the button falls back to "Generating variants" during what is really transcription.
4. **Waking and transcribing at once:** `uploadPhase === "waking"` and `inBrowser === true` both hold. Two mutable locals then decide which status text wins.

## Decision

A framework-free domain layer lives in `frontend/src/domain/`, with no React import anywhere in it.

### Stages (`src/domain/pipeline.ts`)

The machine is generic over the artifact a run produces, so the domain never imports `TranscriptionResult` or `VariantsResponse` from `app/`: `/analyse` uses `PipelineState<Transcription>`, `/generate-variants` uses `PipelineState<VariantsResponse>`.

```ts
type AudioSource =
  | { kind: "upload"; file: File; stored: StoredUpload }   // already on the backend
  | { kind: "recording"; file: File }                      // not uploaded yet
  | { kind: "session"; sourceName: string };               // last upload in the session store

type TranscribeStep =                    // what the status line says, before copy
  | { kind: "starting" } | { kind: "uploading"; phase: "uploading" | "waking" }
  | { kind: "stillWorking" } | { kind: "browser"; step: number /* 0–10 */ }
  | { kind: "savingRetry" } | { kind: "submitRetry" }
  | { kind: "queued"; slow: boolean } | { kind: "processing"; slow: boolean }
  | { kind: "warmingUp" } | { kind: "restarting" };

type PipelineState<A> =
  | { status: "idle" }
  | { status: "transcribing"; runId; source; path: "browser" | "server";
      upload: UploadPhase | null; step: TranscribeStep }
  | { status: "generating"; runId; source }
  | { status: "completed"; runId; source; artifact: A }
  | { status: "failed"; runId; source; error: DomainError; dismissed: boolean };
```

`path` and `upload` replace the two closure flags (`inBrowser`, `uploadPhase`) that used to decide which status copy may overwrite which. They're now state, so the gating rules are reducer cases with unit tests.

Clip analysis on `/analyse` was already a discriminated union (`idle | loading | ready | error`) and is left as it was.

### Errors (`src/domain/errors.ts`)

```ts
type DomainErrorCode =
  | "invalid_input" | "busy" | "warming_up" | "restarting" | "job_lost" | "failed";
type DomainError = { code; message: string; retryable: boolean; retryAfterSeconds?: number };
```

Two pure classifiers, because the two pages have always worded failures differently:

- `toGenerationError(unknown)`: the 429/`x-error-code`/502/503/network logic formerly inlined in `generate-variants/page.tsx`. A restart gets the fixed reassuring copy.
- `toTranscriptionError(unknown)`: `job_lost` for `TranscribeJobLostError`, otherwise `failed` with the raw message (what `/analyse` always showed, even for a 503).

App errors are recognised structurally (`name`, `status`, `code`), so the domain imports nothing from `app/lib`. `isCancellation` covers `RetryCancelledError`, `TranscribeJobSupersededError` and `AbortError`. Only `invalid_input` is non-retryable.

### Transitions (`transition(state, event)`)

Pure, exhaustive `switch`, `assertNever` default: an unhandled event is a compile error.

**Stale-run rule:** every event except `start` and `dismissError` carries a `runId`; an event whose `runId` isn't the current run's (or that arrives while idle) is ignored. This one rule replaces the 9 `requestId`/`mounted` checks.

| From | Event | To |
|---|---|---|
| any | `start{runId, source}` | `transcribing` (path browser, step `starting`) |
| `transcribing` | `browserProgress{fraction}` | step `browser`, rounded down to 10% steps, clamped at 10; same step → same state object |
| `transcribing` | `savingRetry` / `submitRetry` | that step |
| `transcribing` | `fellBack` | path `server`; step unchanged (the last browser line stays on screen, as before) |
| `transcribing` | `uploadPhase{phase}` | `upload` = phase; step `uploading` only on the server path |
| `transcribing` | `uploaded` | `upload` = null |
| `transcribing` | `stillWorking` (8s timer) | step `stillWorking`, only on the server path and not while waking |
| `transcribing` | `submitted` | step `queued` |
| `transcribing` | `serverStatus{status, elapsedMs, waitingReason}` | `warmingUp` / `queued{slow}` / `processing{slow}` (slow = >8s); other statuses ignored |
| `transcribing` | `restarting` | step `restarting` |
| `transcribing` | `generationStarted` | `generating` |
| `transcribing` \| `generating` | `completed{artifact}` | `completed` |
| `transcribing` \| `generating` | `failed{error}` | `failed` (not dismissed) |
| `transcribing` \| `generating` | `cancel` | `idle` |
| `failed` | `dismissError` | `failed`, dismissed (the source stays on screen) |
| `completed` \| `failed` \| `idle` | progress events | unchanged |

Status copy is derived, not set: `describeTranscription(state)` for the `/analyse` spinner and `describeVariantsProgress(state)` for the Variants button (`src/domain/statusCopy.ts`).

### Hooks and pages

`usePipeline<A>()` (`hooks/use-pipeline.ts`) wraps `useReducer(transition)`, owns one `AbortController` per run, aborts the previous run on `start` and the current one on unmount. Pages dispatch events from their existing async I/O (`app/lib/*` unchanged) and render from state. Two pieces of state stay beside the machine on purpose:

- **A pre-run error slot** (`uploadError` on `/analyse`, `notice` on Variants) for errors raised before any run exists: a failed `UploadButton` upload, or "choose a clip first".
- **`result` on Variants**, so a failed regeneration still shows the previous variants under its banner.

## Deliberate departures from the roadmap brief

- **No confidence values.** The brief asks for pitch and chords with confidence. The backend returns neither (`ClipAnalysis` has labels and a histogram only). Inventing a number would be dishonest UI, so `AudioAnalysis` mirrors what the API actually returns. A `confidence` field can be added when the backend emits one.
- **No `uploading` top-level state.** On the real flow, uploading happens *inside* transcription: in parallel with the browser path, and shown only on fallback. A top-level `uploading` state would reorder copy the user sees today. It's modelled as `TranscribeProgress.where === "upload"` instead.
- **"analysing" is spelled `transcribing`.** Pitch extraction is transcription in this codebase (the API route is `/api/transcribe`). Clip analysis is a separate machine.
- **The last variants result is kept outside the run state.** Today a failed regeneration shows the error banner *and* keeps the previous variants on screen. Preserving that (this phase must not change behaviour) means `result` lives beside the machine, not inside `failed`.
- **`src/domain/` is `frontend/src/domain/`.** There was no `src/` before; the domain is the first thing in it.

## Impossible states, after

1. **Half-stored source:** unrepresentable. `AudioSource` is one value; an upload always carries its stored id and filename.
2. **Stale progress:** handled by the stale-run rule, once, in the reducer.
3. **Busy with no description:** the state is now explicit (`transcribing`, step `starting`) and its label is chosen in one place. It still reads "Generating variants" because this phase doesn't change copy. Fixing the wording is now a one-line change in `describeVariantsProgress`.
4. **Waking and transcribing at once:** `path` and `upload` are both state now, and the reducer decides which one shows (the upload phase only on the server path).

## Found while implementing (not changed in this phase)

On `/analyse`, the status line only renders inside the spinner, so the post-run copy the old code set is never shown: "Transcription ready for X.", "We couldn't transcribe that audio clip." and the **job-lost message**. A transcription job lost to a restart ends with the spinner disappearing and nothing on screen. Existing tests assert that "couldn't transcribe" is absent. The machine keeps the `job_lost` code, so surfacing it is a render change in a later phase, not a state change.

## Consequences

- The impossible states 1–4 become unrepresentable or are covered by the stale-run rule.
- Status copy, error classification, and transitions are all pure, so they're unit-tested without React, timers or `fetch`.
- No UI behaviour changes. The existing 257 tests, which pin today's copy and flows, are the regression net. Any test edit in this phase must be a type or import change only, never an expectation change.
- Cost: one more indirection (event → reducer → render) when reading a page's flow. Mitigated by keeping every event name in the user's terms.
