---
description: Phase 7. Build Web Audio and Canvas visualisations, waveform, pitch timeline, piano roll, and confidence metrics.
argument-hint: "[waveform | pitch-timeline | piano-roll | meters]"
disable-model-invocation: true
---

Use the `audio-viz-dev` subagent.

Surface to build: $ARGUMENTS

Shared rules:
- Canvas 2D, one shared `requestAnimationFrame` loop for all animated surfaces, `devicePixelRatio` handled, `ResizeObserver` for sizing.
- Draw from a cached peaks array computed once per zoom level, never from the raw `AudioBuffer` per frame.
- One shared `AudioContext` created on a user gesture. Playback position reads `AudioContext.currentTime`, never a timer counter.
- Full cleanup on unmount: cancel the frame, disconnect nodes, revoke object URLs.
- Every canvas has a text alternative describing what it shows.

Per surface:
- waveform: scrubbable, zoomable, playhead synced to real audio time
- pitch-timeline: detected pitch over time with confidence shown as opacity or height, hover reads out note name and frequency
- piano-roll: generated MIDI notes, note velocity visible, notes also exposed to assistive tech as a list or table
- meters: model confidence and per-stage progress, no meaningless spinners

State the frame cost before and after. Report the numbers.
