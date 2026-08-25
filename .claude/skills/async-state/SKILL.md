---
description: Phase 4. Wire the domain model to TanStack Query and clean up async state, typed API contracts, and cancellation.
argument-hint: "[optional: a specific endpoint or hook]"
disable-model-invocation: true
---

Use the `domain-architect` subagent.

1. Server state moves to TanStack Query. Client-only state (transport position, panel layout, selected track) stays in React state or a small store. Never mirror server data into a store.
2. Every API call gets a typed contract in `src/domain/contracts/`, parsed at the boundary with a schema. No casting a `fetch` result.
3. Every request carries an `AbortSignal`. Cancelling a job actually aborts the request, and the UI returns to `idle`.
4. Query keys are structured and typed, not string concatenation.
5. Retry policy is explicit: retry transient failures with backoff, never retry a validation error. The free-tier backend cold-starts, so the first request needs a longer timeout and a "waking up the server" message rather than a failure.
6. Kill every remaining `any`. Run `npx tsc --noEmit` and fix what it finds.

Do not change visual behaviour. Report the before and after `any` count.

Scope: $ARGUMENTS
