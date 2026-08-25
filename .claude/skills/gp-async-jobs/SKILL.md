---
description: GP2. Replace the blocking transcribe call with an async job plus a persistent jobs table, removing the 30-second analysis cap.
disable-model-invocation: true
---

Read @docs/GUIDED-PASS.md, sections "How I want you to operate", "Context already confirmed", and GP2.

Implement GP2. GP1 must be merged first.

### Storage: decide it yourself from the repo, do not ask

Check for a `DATABASE_URL`, a Neon or Postgres reference in `render.yaml`, the Render
service config, and the backend settings module. Then apply this rule:

- **Postgres is configured** -> use it. SQLite only for local dev, behind the same
  SQLModel layer.
- **Nothing is configured** -> use SQLite, and open the phase report with a warning that
  Render's free web service has an ephemeral filesystem, so the jobs table is wiped on
  every deploy and every hibernate wake, which will break GP3's job-id links in
  production. Name the migration path.

Do not silently pick SQLite because it is faster to wire up. State which branch of the
rule you took and what evidence you took it on.

### Implementation

1. `jobs` table via SQLModel: id, status, result, error, created_at, updated_at.
   No Celery, no Redis queue.
2. `POST /api/transcribe` creates the job, dispatches via FastAPI `BackgroundTasks`,
   returns `{job_id}` immediately.
3. `GET /api/transcribe/{job_id}` returns status and result.
4. Backend first, verified with curl or pytest before any frontend change.
5. Frontend poll loop replacing the blocking call: queued, analysing, done. Reuse the
   existing cold-start retry logic rather than duplicating it, and preserve the existing
   rule that a 504 is not retried. Do not regress that.
6. UI copy states that the timeout is gone, not that analysis got faster. A long clip
   still takes a while on Render's free CPU.

### Handle explicitly

A job whose worker thread is still running when the process restarts. Either reconcile
stale `running` rows on startup or mark them failed with a retry affordance. Say which
you chose and why in the commit body.

Verify: a clip over 60 seconds completes end to end, a page reload mid-job recovers,
`npx prettier --check .` and pytest green. Commit GP2 alone.
