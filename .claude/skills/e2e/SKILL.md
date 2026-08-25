---
description: Phase 3. Write or repair Playwright end-to-end journeys with mocked API responses so tests never call the real model.
argument-hint: "[journey name, or 'all']"
disable-model-invocation: true
allowed-tools: Bash(npm run test:e2e*) Bash(npx playwright*)
---

Use the `e2e-runner` subagent.

Journey to work on: $ARGUMENTS (if empty, cover all five below).

1. Happy path: upload, analyse, generate, play back, export MIDI
2. Rejected file: wrong format and oversized file, both recoverable
3. Backend failure mid-generation, then a successful retry
4. Cancellation mid-generation returns the UI to idle
5. Reload during a running job recovers to a sane state

Rules:
- Intercept the FastAPI routes and serve fixtures from `e2e/fixtures/`. The suite must never invoke inference and must finish under two minutes.
- Select by role and accessible name only. If something cannot be selected that way, fix the component, do not add a test id.
- No `waitForTimeout`. Wait on a condition.
- Tests run in parallel and share no state.

If a test fails, open the page with the Playwright browser tools and look at it before touching the test.
