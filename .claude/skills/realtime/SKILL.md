---
description: Phase 8. Build the real-time generation UX, stage-based progress, SSE or polling, cancellation, and cold-start handling.
argument-hint: "[optional: sse | polling]"
disable-model-invocation: true
---

Long AI jobs are the hardest UX problem in this app. Build for the honest case: the free-tier backend cold-starts and generation takes tens of seconds.

Deliver:
1. A stage tracker driven by the domain state machine, not by a fake timer: uploaded, pitch extracted, chords detected, generating, finalising. Each stage shows what is happening and roughly how long it takes.
2. Live updates via SSE, with polling as the fallback when the connection drops. Reconnect with backoff and no duplicate state.
3. `AbortController` wired end to end. Cancel actually cancels, and the UI returns to `idle` with nothing orphaned.
4. Cold start handled explicitly: if the first request has not responded in a few seconds, say the server is waking up rather than showing a generic spinner.
5. Progress announced to screen readers through a throttled `aria-live="polite"` region. Announce stage changes, not every percentage tick.
6. A job survives a page reload, or the UI recovers cleanly and says what happened.

Write the failing test for each transition first. Use the `tdd-engineer` subagent for the state logic and handle the transport layer yourself.

Transport preference: $ARGUMENTS
