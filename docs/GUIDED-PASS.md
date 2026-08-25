# Guided Improvement Pass (GP1 to GP5)

Numbered GP1 to GP5 deliberately. `docs/ROADMAP.md` has its own Phase 0 to 15, and saying
"start phase 2" with both documents in the repo is ambiguous enough that an agent will
sometimes pick the wrong one.

Relationship between the two: this pass fixes real product problems in the app as it exists.
The roadmap raises the engineering bar around it. Run this pass first. GP2 in particular
delivers what roadmap Phase 8 describes, so doing them in the other order means building
the async UX twice.

## How I want you to operate

Implementation job. Build it, verify it, commit it. No walkthroughs, no diff narration,
no explaining the codebase back to me.

1. **One phase per session.** Do not start the next phase until the current one is
   verified working and committed.
2. **Move code before rewriting it.** Where a phase says "move", the behaviour must be
   equivalent from the user's side. Refactors that were not asked for belong in a
   separate phase.
3. **Verify every phase**: `cd frontend && npm run format:check`, the backend pytest suite, and a
   manual smoke check (dev server up, affected page loads, affected flow completes).
   The Stop hook enforces the first two. The smoke check is on you.
4. **Decide, do not ask, where the repo answers the question.** Where a phase gives you
   a decision rule, apply it and state which branch you took and on what evidence. Stop
   and ask only when the repo genuinely does not settle it and the choice is expensive
   to reverse.
5. **Report, do not tutor.** End each phase with: what changed, what you decided and why,
   what you deliberately did not do, and anything that contradicted the confirmed context.
   Short.
6. **Subagents where they help.** Do not parallelise work that depends on an earlier
   decision. No PR and no push unless I ask.
7. **One commit per phase**, so a phase can be reverted without losing the others.

## Context already confirmed

Treat this section as fact. It was verified against the codebase, so do not re-derive it,
and if you find something here that is no longer true, say so before acting on it.

- `frontend/app/page.tsx` (559 lines) is currently BOTH the landing page and the
  analyse / transcribe / download experience.
- There is no database wired into the codebase today. Persistence is plain files on disk
  with a time-based cleanup loop (`DATA_RETENTION_HOURS`, default 24h), unrelated to the
  30s analysis cap.
- The real "30 second" limit is `MAX_ANALYSIS_DURATION_SEC` in `backend/app/inference.py`,
  set to half of `GENERATION_TIMEOUT_SECONDS` (60s) to leave headroom under Render's
  platform gateway timeout (~100s, not controlled by our env vars). `_run_generation`
  already runs work in a thread pool and only wraps it with `asyncio.wait_for`, so the
  thread keeps running past that timeout regardless.
- Mood detection is a hardcoded heuristic (`heuristic_mood_from_metrics` in
  `backend/app/model/colab_parity.py`) using only tempo, key, and average pitch. Likely
  why neutral songs get labelled sad or happy.
- Text-only "Uploading..." loading states exist in `upload-button.tsx` and
  `generate-variants/page.tsx`. There is no shared spinner component yet.
- Training notebook: `backend/melody_generation_ORDERED_FINAL_(1).ipynb`.
- The frontend already has cold-start retry logic around transcription that deliberately
  skips retrying on a 504. Any new async flow must respect that, not duplicate it.

---

## GP1 — Split the landing page from the analyse page
Command: `/gp-split`

New landing page at `/`: what the app does, a link to the Colab notebook, an about and
user-guide section. Move the current analyse / upload / transcribe / download UI to `/analyse`.
Fix internal nav links.

**Gate:** both routes load, every existing flow still completes from `/analyse`,
`cd frontend && npm run format:check` clean, one commit.

---

## GP2 — Remove the 30-second analysis cap (async job + persistent store)
Command: `/gp-async-jobs`

Researched (a) raise the constants versus (b) async background job with a store.
**Decision: (b), kept minimal.** Raising constants alone still hits Render's ~100s gateway
timeout, so it nudges the cap rather than removing it. (b) is the only option that fixes it.

- One `jobs` table: id, status, result, error, timestamps. SQLModel or SQLAlchemy.
  No Celery and no Redis queue, that is overkill for this.
- `POST /api/transcribe` creates a job, runs analysis via FastAPI `BackgroundTasks`,
  returns `{job_id}` immediately.
- New `GET /api/transcribe/{job_id}` for polling status and result.
- Frontend: poll loop replacing the blocking call, with real states (queued, analysing,
  done), reusing the existing retry instincts rather than duplicating them.
- Copy must be honest: this fixes the *timeout*, not the *speed*. A long clip still takes
  a while on Render's free CPU.
- This table also backs GP3's download page.

**Storage rule, applied by the agent, not asked about:** check for a `DATABASE_URL`, a
Neon or Postgres reference in `render.yaml`, the Render service config, and the backend
settings module. If Postgres is configured, use it, with SQLite for local dev behind the
same SQLModel layer. If nothing is configured, use SQLite and open the phase report with
a warning: Render's free web service has an ephemeral filesystem, so the table is wiped on
every deploy and every hibernate wake, which breaks GP3's job-id links in production.
State which branch of the rule was taken and on what evidence.

**Gate:** a clip longer than 60s completes end to end, a mid-job page reload recovers,
pytest green, one commit.

---

## GP3 — Move MIDI download to its own page
Command: `/gp-download-page`

A dedicated download and result page referencing results by job id, not by re-uploading.
Mind the ~4.5 MB Vercel body limit. Replaces the inline download buttons in the analyse
page and in `generate-variants/page.tsx`.

**Gate:** a job-id URL opens the result cold in a new tab, an expired or unknown id shows
a real empty state rather than a crash, one commit.

---

## GP4 — Improve the Colab notebook
Command: `/gp-notebook`

Independent of GP1 to GP3. Touches only the notebook and `colab_parity.py`, so this one
can run in a parallel session on its own branch.

- Evaluate Claude API integration options for the notebook (data augmentation, labelling,
  output review) as a written comparison in `docs/notebook-claude-api.md`, ending in a
  recommendation. A document, not a conversation.
- Fix the mood heuristic by extending its features, not by training a classifier. A
  classifier adds a training dependency and a reproducibility burden to a thesis artifact
  for accuracy we cannot yet measure. Record it as a rejected option in `docs/adr/`.
- Investigate and propose concrete, explainable improvements to CVAE and IDDM-PPO training
  quality. Not a rewrite.

**Gate:** the notebook still runs top to bottom, mood output is sane on a set of clips
I pick, one commit.

---

## GP5 — Shared spinner component
Command: `/gp-spinner`

Depends on GP2, since the polling state is one of the call sites.

Explore-agent audit of every text-only loading state (confirmed: `upload-button.tsx`,
`generate-variants/page.tsx`, plus GP2's polling state, plus whatever else turns up).
Build one reusable Tailwind `<Spinner />` matching the existing purple and dark theme,
and swap every call site. Delete the text-only states rather than leaving both.

**Gate:** no text-only loading state left, the spinner respects `prefers-reduced-motion`,
one commit.

---

## Dependency order

```
GP1 -> GP2 -> GP3 -> GP5        sequential, each needs the one before
GP4                             independent, safe to run in a parallel session
```
