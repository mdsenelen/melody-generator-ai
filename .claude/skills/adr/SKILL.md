---
description: Phase 13. Write an architecture decision record for a non-obvious technical choice, grounded in this codebase.
argument-hint: "<decision, e.g. 'SSE over polling'>"
disable-model-invocation: true
---

Use the `adr-writer` subagent to write an ADR for: $ARGUMENTS

Before writing, read the code the decision actually touches, and pull the real numbers where they exist (bundle size, request count, frame cost, response time). An ADR with no measurement and no rejected alternative is not worth committing.

Write to `docs/adr/NNNN-slug.md` using the next free number, and add a row to the index in `docs/adr/README.md`.

Decisions this project still owes: SSE versus polling versus WebSockets; Canvas versus SVG for the waveform and piano roll; TanStack Query versus a store for server state; why E2E mocks inference; how the free-tier cold start is handled in the UI.
