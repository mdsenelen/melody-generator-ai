# Progress log

Appended by `/phase-commit` at the end of each phase. Newest last.

| Phase | Date | Shipped | Deferred | Needs an ADR |
|---|---|---|---|---|
| GP5 spinner | 2026-08-26 | One reusable `<Spinner>` (accessible name via `aria-labelledby` on `role="status"`, `motion-reduce:animate-none`) swapped into every loading call site: `upload-button`, `generate-variants`, `chord-graph`, analyse's job-polling state (now surfacing the real `statusMessage`, previously computed but unrendered), and `listen-progressions`' unlabelled `animate-spin` div. All bespoke loading UI (bounce-bar animation, raw spinner div, text-swap buttons) deleted, not left alongside the new component. | `audio-recorder`'s "Recording..." status text (user-controlled state with a Stop button, not a passive wait — a spinner would misrepresent it); `chord-dropdown`'s silent no-indicator fetch (no existing loading UI there to swap, out of scope). | none |
