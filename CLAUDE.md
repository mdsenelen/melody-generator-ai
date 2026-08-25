# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full-stack AI music generation app. Users upload or record audio; the backend transcribes it, analyzes pitch/chords/mood/key, and generates new melodies using trained PyTorch models (CVAE + IDDM-PPO). Output is MIDI or WAV, playable in the browser.

This started as a thesis project and is now being rebuilt into a production-grade portfolio piece serving real users — see the `architecture-reference` skill for what that means concretely (non-functional requirements) and what's still missing to get there (production readiness), and `docs/ROADMAP.md` for the frontend engineering plan that's driving the rebuild.

## Commands

### Frontend (Next.js — run from `frontend/`)

```bash
npm run dev      # Dev server on http://localhost:3000
npm run build    # Production build
npm run start    # Production server
npm test         # jest + RTL (npm test -- --ci in CI)
npm run typecheck  # tsc --noEmit
npm run format     # prettier --write .
npm run format:check
```

No `npm run lint` or `npm run test:e2e` yet — no ESLint config or Playwright test runner exists in `frontend/` today (see "Open questions").

### Backend (FastAPI — run from `backend/`)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Environment required: `PYTHONPATH=.` (set in `backend/.env`).

This alone is enough locally: the dev server also runs the transcription worker on a background thread (see the `architecture-reference` skill's "Async Transcription Job Workflow"), so no second process is needed. Only run the worker separately if you've deliberately set `RUN_WORKER_IN_PROCESS=false` to mirror production:

```bash
python -m app.worker_main
```

### Tests (backend only)

```bash
cd backend
pip install -r requirements-dev.txt
pytest                                          # all tests
pytest tests/test_inference_variants.py        # inference + model loading
pytest tests/test_jobs.py -v                   # async job workflow
```

## Architecture reference

Full backend/frontend architecture, model checkpoint schemas, async job workflow internals, key configuration, deployment, and the production-readiness checklist live in the `architecture-reference` skill (`.claude/skills/architecture-reference/SKILL.md`) rather than here, to keep this file cheap to load every turn. That content changes fast — three of its "gap" bullets went stale in the same session it was split out of this file, so treat it as a pointer to verify against, not a frozen contract.

Load-bearing facts worth keeping inline because nearly every task touches them:

- Two model stacks exist; only `model/colab_parity.py` (MelodyCVAE + IDDM-PPO) is active. `model/vae.py` and `model/utils.py` are legacy, not used at inference time.
- Transcription is async and job-based (`backend/app/jobs/`), not synchronous — `POST /api/transcribe` returns 202 + a job id immediately; there is no synchronous transcribe route anymore.
- `RUN_WORKER_IN_PROCESS` defaults to `true` for local dev; production sets it `false` and runs `python -m app.worker_main` as a separate service.
- The production job stack (Neon Postgres, Render Redis, Backblaze B2) is live; per-IP rate limiting (`app/rate_limit.py`) and a request body size cap (`app/request_limits.py`) are both live too, on `/api/transcribe`, `/api/generate-variants`, and (for the body cap) every route.
- Free-tier constraints are real, not theoretical: the web service's 512MB memory ceiling means it currently can't reliably serve two transcriptions back-to-back without an OOM restart — see the skill's "Cost constraints" section before assuming concurrency is safe.

## Scope rules

- The frontend is the surface being upgraded. Do not refactor Python model code (`backend/app/model/`, the notebooks) unless a roadmap phase calls for it.
- Never commit audio files, model weights, soundfonts, `.env*`, notebook outputs, or anything over 5MB.
- Never rewrite git history, force-push, or `git add -A` without being asked.
- If a change touches the FastAPI contract, update the shared TypeScript types in the same commit.

## Non-negotiables (every task, every phase)

1. TypeScript strict, zero `any` (already enforced — `frontend/tsconfig.json` has `"strict": true`). Model status via discriminated unions, never loose booleans.
2. No new behaviour without a test. Write the failing test first, then the implementation.
3. Every async operation is cancellable (`AbortController`) and has explicit idle/loading/error/empty states.
4. Accessibility is part of "done": keyboard path, visible focus, ARIA live region for status changes, `prefers-reduced-motion` respected.
5. No secrets in client code. Validate uploads for MIME, extension, and size on the client; the server already revalidates size (`MAX_UPLOAD_BYTES` middleware) and duration (`MAX_ANALYSIS_DURATION_SEC`).
6. Canvas and audio analysis never block the main thread. Heavy buffer work goes to a Worker or AudioWorklet.
7. Error boundaries are scoped. A crashing audio player must not take down the workspace. (`react-error-boundary` is already a dependency; no top-level boundary is wired yet.)

## Definition of done

- `npm run typecheck` and `npm test -- --ci` pass (both are CI gates today, `.github/workflows/ci.yml`). `npm run lint` once ESLint is configured — not a current gate.
- Behaviour covered at the right level: unit for logic, RTL for components, Playwright for real user journeys once E2E lands (roadmap phase 3, `/e2e`) — no Playwright test runner is wired in yet, only the `playwright` MCP server for interactive browser driving.
- No console errors or React warnings in the browser.
- One logical commit with a conventional-commits message.

## Release gate (phases touching the API contract, persistence, or object storage)

Applies to GP2, GP3, and roadmap Phase 8. Not to every commit. The per-change
Definition of Done above still applies on top of this.

- Frontend and backend deploy separately, so a contract change ships either
  backward-compatible or in two ordered steps. State which, in the commit body.
- Migrations run forward against both a fresh database and the current
  production schema.
- No change to an artifact path or bucket key format without a fallback that
  reads the old format. Existing jobs must keep resolving.
- Verify against the deployed app, not just localhost, before closing the phase.

## Phase discipline

Two programmes, run in this order:

1. **`docs/GUIDED-PASS.md`** — GP1–GP5, product fixes to the app as it exists (commands `/gp-*`). Run first: GP2 delivers what roadmap Phase 8 describes, so doing the roadmap first means building that async UX twice.
2. **`docs/ROADMAP.md`** — Phase 0–15, the engineering bar, each with a slash command in `.claude/skills/` and a prompt in `docs/PHASE-PROMPTS.md`.

One phase per session. Build, verify, commit, report — no diff narration, no explaining the codebase back. Where a phase gives a decision rule, apply it and state which branch you took; stop and ask only when the repo genuinely doesn't settle the question and the choice is expensive to reverse. At the end of a phase, append what changed to `docs/PROGRESS.md` via `/phase-commit`.

A Stop hook (`.claude/hooks/verify-gate.sh`) mechanically blocks ending a turn while `prettier --check` or `tsc --noEmit` are red on changed files (pytest only runs under `GATE_PYTEST=1`, since it's slow). Escape hatch: `touch .claude/skip-gate` — delete it to re-arm.

## House style

- Tests live under `frontend/__tests__/<category>/` (e.g. `__tests__/components/`, `__tests__/lib/`, `__tests__/pages/`), mirroring the source tree — this is the established convention (9+ existing test files); don't switch to co-located `Foo.test.tsx` files next to source.
- Domain logic will move to `src/domain/` (framework-free, directly unit-testable) once roadmap phase 1 (`/domain-model`) lands — today it's intermixed with `app/`, `hooks/`, and `utils/`, since that phase hasn't run yet.
- Components receive data, they do not fetch it. Data access lives in hooks.
- Name things by what the user controls ("Generate melody"), not how the system works ("Run inference").
- No barrel files that re-export half the app.

## Open questions

- The installed roadmap/skill package (`.claude/skills/*`, `.claude/hooks/format-changed.sh`) consistently assumes `pnpm`, but this repo uses `npm` (`package-lock.json`, no pnpm/yarn lockfile). `format-changed.sh` also calls `eslint`, which isn't installed. Both mean the `PostToolUse` formatting hook currently no-ops (fails silently) on every edit. Not fixed as part of this merge — flagging for a deliberate pass, since it touches many files outside CLAUDE.md/`.mcp.json`/`.claude/settings.json`.
- Is adding ESLint a standalone task now, or intentionally deferred to roadmap phase 12 (`/ci`)? "Definition of done" above assumes the latter.
