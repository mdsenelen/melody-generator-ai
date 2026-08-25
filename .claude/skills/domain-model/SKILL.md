---
description: Phase 1. Replace boolean state flags with an explicit typed domain model and job state machine for the audio to melody pipeline.
argument-hint: "[optional: a specific flow to model first]"
disable-model-invocation: true
---

Use the `domain-architect` subagent.

Model the pipeline as four typed stages: `AudioInput`, `AudioAnalysis` (pitch and chords, each with a confidence value), `MelodyGeneration`, `GeneratedArtifact` (MIDI and WAV).

Then replace every `isLoading` / `isProcessing` / `isError` cluster with one discriminated union covering `idle`, `uploading`, `analyzing`, `generating`, `completed`, and `failed`. `failed` carries a `DomainError` with a code, a user-facing message, and a `retryable` flag.

Requirements:
- Put the domain in `src/domain/`, framework-free, no React import anywhere in it.
- Write the transition function as pure code with an exhaustive `switch`, so an unhandled state is a compile error.
- Do not change UI behaviour in this phase. This is a type and state refactor. The app must behave identically when you finish.
- `npm run typecheck` must pass with zero errors and zero new `any`.

Write the union and the transition table into `docs/adr/` as part of this phase, then
implement against it. State any impossible state the old code allowed.

Start with: $ARGUMENTS
