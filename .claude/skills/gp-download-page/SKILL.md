---
description: GP3. Move MIDI download to a dedicated result page addressed by job id, replacing the inline download buttons.
disable-model-invocation: true
---

Read @docs/GUIDED-PASS.md. GP1 and GP2 must be merged.

Implement GP3.

1. Audit every inline download button. Confirmed: the analyse page and
   `generate-variants/page.tsx`. Grep for others before assuming that list is complete.
2. Build a result page addressed by job id, reading from the GP2 jobs table. Never
   re-upload the audio: the Vercel body limit is ~4.5 MB and the file is already on the
   server.
3. Cover all four states: running, succeeded, failed, and unknown-or-expired past
   `DATA_RETENTION_HOURS`. An expired job must read as expired, not as an error.
4. Replace the inline buttons with links to the result page. Remove the old download
   code paths, do not leave both.

Verify: a job-id URL opens cold in a new tab and serves the result. A fabricated id shows
the empty state, not a crash or a blank page. `cd frontend && npm run format:check` and pytest green.
Commit GP3 alone.
