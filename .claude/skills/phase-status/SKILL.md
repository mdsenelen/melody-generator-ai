---
description: Show roadmap progress, which phase is current, what the last phase delivered, and what the next concrete step is.
allowed-tools: Read Grep Glob Bash(git log*) Bash(pnpm *)
---

## Repo signal
- Recent commits: !`git log --oneline -15 2>/dev/null || true`
- Progress log: @docs/PROGRESS.md
- Roadmap: @docs/ROADMAP.md

## Task
Report in under 20 lines:
1. Which phases look complete based on what is actually in the repo, not on what the log claims.
2. Which phase is in progress and what is left in it.
3. The single next action, specific enough to start immediately.
4. Any gate currently failing (typecheck, lint, tests).

If the progress log disagrees with the repository, trust the repository and say so.
