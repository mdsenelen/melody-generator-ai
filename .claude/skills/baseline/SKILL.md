---
description: Phase 0. Audit repository state, security risks, and repo hygiene, then produce a baseline report. Run this before starting any other phase.
argument-hint: "[optional area to focus on]"
disable-model-invocation: true
allowed-tools: Read Grep Glob Bash(git *) Bash(npm *) Bash(npx *) Bash(du *) Bash(find *)
---

## Current state
- Branch: !`git branch --show-current 2>/dev/null || true`
- Status: !`git status --short 2>/dev/null | head -30 || true`
- Tracked files over 5 MB: !`git ls-files -z 2>/dev/null | xargs -0 du -h 2>/dev/null | sort -rh | head -10 || true`

## Task
Delegate to the `repo-auditor` subagent to produce the full baseline, then act on it yourself.

After the audit returns:
1. Write the report to `docs/BASELINE.md` with today's date.
2. Fix only the hygiene items that are safe and mechanical: extend `.gitignore`, untrack files that should never have been committed (use `git rm --cached`, never `git filter-repo`), and remove dead scripts.
3. For anything that needs history rewriting or a credential rotation, stop and list it under "Requires a human decision". Do not act on it.
4. Add the npm scripts the roadmap depends on if they are missing: `typecheck`, `lint`, `test`, `test:e2e`, `analyze`.
5. Commit as `chore: baseline audit and repo hygiene`.

Then print the three highest-risk items and ask which to tackle first.

Focus area if given: $ARGUMENTS
