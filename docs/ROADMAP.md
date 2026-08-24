# melody-generator-ai: frontend engineering roadmap

Goal: without rewriting the AI/music system, turn it into a production-grade portfolio
piece that demonstrates React and TypeScript at senior level, AI product UX, technical
(DAW-style) UI, Web Audio, real-time visualisation, TDD with Playwright, and CI/CD.

Each phase has a command in `.claude/skills/` and a prompt in `PHASE-PROMPTS.md`.

| # | Phase | Command | Gate |
|---|---|---|---|
| 0 | Safety, cleanup, baseline | `/baseline` | `docs/BASELINE.md`, clean tree, nothing over 5 MB tracked |
| 1 | Architecture and domain modelling | `/domain-model` | Zero `any` in `src/domain/`, behaviour unchanged |
| 2 | Testing foundation and TDD | `/tdd` | Every critical path covered, each test seen failing first |
| 3 | Playwright E2E and mock strategy | `/e2e` | Five journeys green in under two minutes |
| 4 | React, TypeScript, async state | `/async-state` | Zero `any`, cancellation aborts real requests |
| 5 | UI system and accessible primitives | `/ui-kit` | No raw hex outside tokens, every primitive keyboard-operable |
| 6 | Audio workspace (DAW layout) | `/workspace` | Layout persists, mobile works, panels fail independently |
| 7 | Canvas and Web Audio visualisation | `/audio-viz` | 60 fps playback, no leaked AudioContext |
| 8 | Async and real-time UX | `/realtime` | Cancel aborts, cold start explained, reload recovers |
| 9 | Performance and Web Workers | `/perf` | Under 250 kB first load, no long task over 50 ms |
| 10 | Accessibility | `/a11y` | Zero Critical or Serious, full keyboard path |
| 11 | Security and reliability | `/harden` | Zero Critical or High, boundaries proven |
| 12 | CI/CD | `/ci` | Green on a PR, broken PR actually blocked |
| 13 | Documentation and ADRs | `/adr` | Five ADRs, each with a real alternative and a real number |
| 14 | Portfolio presentation | `/showcase` | Understandable by a stranger in 60 seconds |
| 15 | Multi-agent senior review | `/senior-audit` | PASS in every category, blocking list empty |

## Domain pipeline

```
AudioInput -> AudioAnalysis (pitch, chords, confidence) -> MelodyGeneration -> GeneratedArtifact (MIDI, WAV)
```

## Job state machine

```
idle -> uploading -> analyzing -> generating -> completed
                \         \           \
                 +---------+-----------+--> failed (retryable | terminal)
                                        \
                                         +--> idle (cancelled)
```

## Workspace layout target

```
+-------------------------------------------------------------+
| Header: track title | transport | export MIDI/WAV | settings |
+------------------------------+------------------------------+
| Audio input and controls     | Waveform visualiser          |
+------------------------------+------------------------------+
| Pitch and chord analysis     | Generated MIDI piano roll    |
+------------------------------+------------------------------+
```

## Dependency order

0 to 4 are sequential and everything else depends on them. 5 must precede 6. 7 must precede
9. 12 must precede 15. If time runs short, cut 9 and 14 before cutting 2 and 10.
