---
description: Phase 10. Audit accessibility against WCAG 2.2 AA and fix the findings, including screen-reader support for audio status and canvas surfaces.
argument-hint: "[optional: a component or route]"
disable-model-invocation: true
---

Run the `a11y-auditor` subagent first, then fix what it reports, highest severity first.

Scope: $ARGUMENTS (if empty, audit the whole workspace route).

Specific to this app:
- Generation status must reach a screen reader through a throttled `aria-live="polite"` region. Failures use `role="alert"`.
- The waveform canvas needs a text alternative: duration, detected key, tempo, and current playhead position.
- The piano roll must expose its notes to assistive tech as a list or table. Canvas alone is invisible.
- Custom sliders (transport scrub, generation parameters) implement the full APG slider keyboard pattern: arrows, Home, End, PageUp, PageDown, and `aria-valuetext` that reads as time or a note name, not a raw number.
- Confidence, state, and selection each need a second channel beyond color.
- `prefers-reduced-motion` disables the playhead animation and panel transitions.

After fixing, re-run the auditor and show the before and after finding counts. Add a regression test for each Critical fix.
