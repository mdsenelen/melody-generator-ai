---
description: Phase 15. Run the full multi-agent senior engineering review across architecture, types, tests, E2E, accessibility, security, and performance, and return a PASS, WARN, or FAIL verdict.
argument-hint: "[optional: base branch, default main]"
disable-model-invocation: true
context: fork
agent: release-gatekeeper
background: false
---

## Diff under review
- Base: $ARGUMENTS (default `main`)
- Changed files: !`git diff --name-only main...HEAD 2>/dev/null | head -60 || true`
- Stat: !`git diff --stat main...HEAD 2>/dev/null | tail -5 || true`

## Task
Run the full audit. Fan out to `repo-auditor`, `a11y-auditor`, `security-reviewer`, `perf-analyst`, and `e2e-runner` in parallel, deduplicate their findings, and return the verdict block.

Judge against the project bar: TypeScript strict with zero `any`, critical paths covered by tests, all five E2E journeys passing, WCAG 2.2 AA, no Critical or High security findings, and the performance budgets in `CLAUDE.md`.

Be strict. If you would not put this in front of a senior hiring manager, it is not a PASS.
