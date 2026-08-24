---
description: Close out the current phase, verify the gates, update the progress log, and commit.
argument-hint: "<phase number and name>"
disable-model-invocation: true
allowed-tools: Bash(pnpm *) Bash(git add*) Bash(git commit*) Bash(git status*) Bash(git diff*) Read Edit Write
---

## Gates
- typecheck: !`pnpm typecheck 2>&1 | tail -5 || true`
- lint: !`pnpm lint 2>&1 | tail -5 || true`
- tests: !`pnpm test --silent 2>&1 | tail -8 || true`
- staged: !`git status --short || true`

## Task
Closing phase: $ARGUMENTS

1. If any gate above failed, stop and fix it before anything else. Do not commit on red.
2. Append an entry to `docs/PROGRESS.md`: the phase, the date, what shipped, what was deliberately deferred, and any decision that needs an ADR.
3. Stage only the files that belong to this phase. Never `git add -A`.
4. Commit with a conventional-commits message summarising the phase.
5. Print the diff stat and name the next phase.

Do not push.
