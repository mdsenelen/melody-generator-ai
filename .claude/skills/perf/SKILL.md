---
description: Phase 9. Measure and fix frontend performance, bundle size, main-thread blocking, and Worker offloading.
argument-hint: "[optional: bundle | render | audio]"
disable-model-invocation: true
allowed-tools: Bash(npm run build*) Bash(npm run analyze*)
---

Use the `perf-analyst` subagent to measure first, then use `audio-viz-dev` to fix what it finds.

Do not change a line before there is a number.

Budgets:
- First-load JS for the workspace route under 250 kB gzipped
- No long task over 50 ms during playback
- 60 fps on the waveform while a generation job is running

Fix in this order:
1. Move buffer analysis (peaks, RMS, pitch histogram) into a Worker with transferable buffers. If it runs in the render path, use an AudioWorklet.
2. Code-split anything not needed on first paint: piano roll, export pipeline, analysis view.
3. Remove render-loop waste: context providers holding fast-changing values, derived arrays rebuilt every frame, list rendering without virtualisation.
4. Trim dependencies over 50 kB gzipped that are not on the critical path.

Report the before and after number for every change. A change with no measured improvement gets reverted.

Focus: $ARGUMENTS
