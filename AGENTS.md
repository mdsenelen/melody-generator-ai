# AGENTS.md

Standing instructions and guardrails for AI coding agents (Claude Code, Codex, or any other agent) working in this repository. `CLAUDE.md` is the source of truth for architecture, commands, and conventions — read it first. This file is about *how to behave* while working here, not what the system does.

This is a **production-grade project serving real users**, not a thesis demo — see `CLAUDE.md`'s "Non-Functional Requirements" and "Production Readiness" sections before touching anything security-, cost-, or availability-sensitive. Treat every change as something a real user could be affected by, not a local experiment.

---

## Definition of done

- **Run the relevant tests before considering any task finished**, and don't report success without having actually run them:
  - Backend: `cd backend && pytest` (install `requirements-dev.txt` first if needed).
  - Frontend: `cd frontend && npm run typecheck && npm test -- --ci`.
  - If you touched formatting-sensitive files, also run `npx prettier --write .` from the repo root (or `npm run format:check` to verify without writing) — CI enforces this and will fail on drift.
- Match what CI actually checks (`.github/workflows/ci.yml`): frontend format-check + typecheck + test + build, backend pytest. If your change would fail any of those, fix it before calling the task done — don't leave it for CI to catch.
- For frontend/UI changes, start the dev server and exercise the feature in a browser if you're able to. Passing tests verify correctness, not that the feature actually works end-to-end — say explicitly if you couldn't verify this rather than assuming it's fine.
- Don't mark a task complete with known-broken tests, type errors, or formatting drift "left for later." If you can't fix something in scope, say so explicitly instead of silently shipping it.

## Secrets and credentials

- **Never commit secrets.** `.env`, `.env.local`, and friends are gitignored — keep it that way. Before staging or committing, check `git status`/`git diff` for anything that looks like a credential, API key, or connection string, even in a file whose name looks innocuous.
- Backend/frontend env vars are configured per-service in the Render/Vercel dashboards (no `render.yaml` in this repo, by existing convention) — don't invent a checked-in secrets file as a shortcut.
- If a task seems to require a real credential (API key, DB password, cloud storage key) that isn't already available in the environment, stop and ask rather than fabricating a placeholder that looks real or hardcoding one "temporarily."

## Infrastructure and cost

- **Ask before provisioning or modifying any paid infrastructure** — a new Render service, Postgres/Key Value instance, S3/R2 bucket, upgrading a plan tier, or anything else that costs money or is hard to reverse. This project intentionally runs on free/cheap tiers right now (see `CLAUDE.md`'s "Cost constraints"); don't move it off that without the user explicitly deciding to.
- Don't push to `main`-deployed branches, trigger a Render/Vercel deploy, or run destructive git operations (`push --force`, `reset --hard`, deleting branches) without explicit confirmation for that specific action. This repo auto-deploys on push to `main` (see "Production Readiness" in `CLAUDE.md`) — a push there is a production deploy, not a local action.
- If you're investigating live infrastructure (e.g. via the Render MCP tools), read-only calls are fine; anything that creates, updates, or deletes a resource needs explicit sign-off first.

### Active exception: async job system verification/deploy (current task)

Scoped exception to the "don't push / don't deploy without asking each time" rule above, granted for the specific task of verifying and deploying the async transcription job system (Postgres/Neon, Redis, B2 wiring). While this task is active, pushing to `origin/main` and iterating through multiple resulting deploys does **not** need per-step confirmation, as long as all of the following hold:

1. Only already-reviewed code is being deployed — the two existing commits (`4b1e5d6`, `9aaee67`) or bug fixes to them found during verification (e.g. the `_init_schema` Postgres-transaction fix). Not new, unreviewed features.
2. Report back after each deploy with what happened, rather than silently chaining deploys one after another.
3. As of 2026-08-24, the live authorization is narrower than "up to 3 attempts": only the first deploy of this task is pre-authorized. Ask before any *additional* deploy beyond that first one, and ask immediately if anything found looks like a deeper architectural problem rather than a small bug — don't wait for a 3rd attempt to raise it. (The "up to 3 attempts" framing above was the original draft of this exception; the user tightened it in practice to "ask before deploy #2" for this task's actual execution. Loosen back only if the user says to.)
4. Everything else is unchanged: provisioning or modifying any paid infrastructure, and any change outside this task's scope, still needs explicit sign-off first — this exception only covers pushing/deploying already-reviewed code for this task.

This is a standing pattern worth keeping in mind generally: routine debugging within an already-approved task doesn't need per-step confirmation, but scope changes and paid infra always do.

Remove this section once the async job system is verified and stably deployed — it's scoped to this task, not a permanent loosening of the push/deploy rule above.

## Code conventions

- Follow `CLAUDE.md` exactly — architecture boundaries (e.g. `inference.py` has no import-time dependency on `jobs/`), the async job workflow's failure/retry model (4xx = permanent, 5xx/unexpected = retried with backoff), the two-model-stack distinction (legacy `WebVAE` vs. active `MelodyCVAE`/IDDM-PPO), and existing patterns like the lazy-singleton model/job-adapter loading.
- No comments explaining *what* code does; only ones that capture non-obvious *why* (a hidden constraint, a workaround, a subtle invariant) — matches the existing style throughout this codebase.
- Don't add abstractions, config flags, or "just in case" error handling beyond what the task needs. Don't leave half-finished implementations.
- Prefer editing existing files over creating new ones; don't create documentation files unless asked.

## Production-readiness guardrails

Given this is now meant for real users, be deliberate about anything that touches the gaps already tracked in `CLAUDE.md`'s "Production Readiness" section:

- Don't quietly work around the lack of auth/rate limiting (e.g. by adding a client-side-only guard) — either leave it as a known gap or raise it explicitly if a task depends on it being closed.
- If a change increases exposure (widens CORS, removes a size/duration cap, adds a new fully-public endpoint that triggers real inference cost), call it out explicitly rather than folding it in silently.
- If you notice the dev-only fallback stack (SQLite / in-process queue / local disk / in-process worker thread — see `CLAUDE.md`'s "Async Transcription Job Workflow") being relied on in a way that assumes production behavior, flag the mismatch instead of assuming it's fine.

## When in doubt

Match the general operating principle: local, reversible actions (editing files, running tests, reading logs) don't need a check-in. Anything hard to reverse, costs money, or is visible to other people (a deploy, a paid resource, a force-push, a message sent externally) does. If a task is ambiguous about scope — especially around the guardrails above — ask rather than guessing.
