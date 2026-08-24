---
description: Phase 12. Set up the GitHub Actions pipeline with quality gates, lint, typecheck, unit tests, Playwright, and build.
argument-hint: "[optional: a specific job]"
disable-model-invocation: true
allowed-tools: Read Grep Glob Edit Write Bash(gh *) Bash(pnpm *)
---

Build `.github/workflows/ci.yml`.

Jobs, running in parallel where possible, all on pull request and on push to main:
1. `quality`: install with a frozen lockfile, `pnpm lint`, `pnpm typecheck`
2. `test`: `pnpm test --coverage`, upload the coverage summary as an artifact
3. `e2e`: `pnpm exec playwright install --with-deps chromium`, then `pnpm test:e2e` against fixtures only, with the trace and HTML report uploaded on failure
4. `build`: `pnpm build`, and fail the job if the workspace route exceeds its 250 kB gzipped first-load budget

Requirements:
- Cache the pnpm store and the Playwright browsers. A cold run should still be under six minutes.
- No job may reach the real model, the real backend, or any secret. Everything runs against fixtures.
- Concurrency group per branch, cancelling in-progress runs.
- Add a branch protection note to the README listing which checks must pass before merge.

After writing the workflow, validate the YAML and dry-run each command locally so CI does not fail on something we could have caught here.

Job: $ARGUMENTS
