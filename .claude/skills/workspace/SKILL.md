---
description: Phase 6. Build the multi-panel DAW-style audio workspace layout with resizable, collapsible panels.
argument-hint: "[optional: a specific panel]"
disable-model-invocation: true
---

Use the `ui-systems-dev` subagent.

Build the workspace shell:
```
Header:  track title | transport | export MIDI/WAV | settings
Left:    audio input, source controls, generation parameters
Right:   waveform visualiser
Bottom:  pitch and chord analysis grid | generated MIDI piano roll
```

Requirements:
- Panels resize and collapse, and the layout persists across reloads.
- The whole layout collapses to a single stacked column on mobile with no loss of function.
- Density reads as a tool, not a landing page: tight spacing, monospaced numerics for time and frequency, color used to carry state and confidence rather than to decorate.
- Panel content is lazily mounted. The piano roll does not load until there is something to show.
- Each panel sits inside its own error boundary, so one crashing panel leaves the rest usable.
- Keyboard: panels are reachable, collapse toggles are buttons, and focus is never lost when a panel closes.

Take one deliberate aesthetic risk you can justify and keep everything around it quiet. Avoid the default AI-app palette.

Panel to work on: $ARGUMENTS
