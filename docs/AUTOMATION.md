# What to automate, and what not to

Four rungs, from most supervision to least. Pick per task, not per project.

| Rung | What it is | Right for | Wrong for |
|---|---|---|---|
| 1. Subagents in a session | Delegation. You still approve every plan. | Everything in GP1 to GP5 | Nothing, this is the default |
| 2. Orchestrated launch | `scripts/run-phase.sh` sets up the branch and opens plan mode | Removing the typing, not the judgement | Skipping the plan review |
| 3. Parallel sessions | Git worktrees, one branch each | GP4, roadmap audits | GP1 to GP3, they are sequential |
| 4. Headless | `claude -p`, no human in the loop | Read-only audits, CI review | Any phase that writes code |

## The one automation worth having above all others

`.claude/hooks/verify-gate.sh` is a Stop hook. It fires when Claude thinks it is done and
refuses to let the turn end while `prettier --check` or `tsc --noEmit` is red. Claude sees
the failure text and has to fix it before it can stop.

This matters more than any agent you could add, because it converts your rule 2 ("do not
break what already works") from a request into a mechanical constraint. A prompt
instruction decays across a long session. A hook does not.

Four things it guards against, all of which are common ways Stop hooks get built wrong:
- **Infinite loops**: it checks `stop_hook_active` and exits 0 if it already blocked once.
- **Firing on conversation turns**: it exits 0 when nothing in the working tree changed,
  so asking a question does not trigger a typecheck.
- **Firing during planning**: it exits 0 when `permission_mode` is `plan`.
- **Its own missing dependencies**: it parses the hook payload in pure bash rather than
  with `jq`, which macOS does not ship. A gate that silently no-ops because a tool is
  missing is worse than no gate, because you think you are protected. If `prettier` or
  `tsc` is not installed it skips that check rather than failing on it.

Escape hatch: `touch .claude/skip-gate`. Delete the file to re-arm it. Use it when a
failure is genuinely pre-existing, not to get past your own change.

Backend tests are opt-in, since they are slow enough to risk the hook timeout:
`GATE_PYTEST=1 claude`.

## Parallel work

You already run parallel sessions on separate branches. Worktrees make that safe, because
each session gets its own checkout and cannot see another session's uncommitted files:

```bash
./scripts/run-phase.sh gp-notebook --worktree
```

Subagents can do the same with `isolation: worktree` in their frontmatter, which gives the
agent a throwaway copy of the repo branched from your default branch. Worth adding to an
agent you want to let run loose.

What can actually run in parallel here:
- GP4 (notebook) alongside anything. It touches only the notebook and `colab_parity.py`.
- Roadmap audits alongside anything. They are read-only.
- Nothing else. GP1 to GP3 to GP5 form a chain, and roadmap phases 0 to 4 are sequential.

Running two write-capable sessions against the same files does not produce two features.
It produces a merge conflict plus two agents each confident they finished.

## Fully unattended

```bash
./scripts/audit.sh full     # multi-agent review, writes to reports/
./scripts/audit.sh a11y     # one auditor
```

Safe unattended because every agent it invokes has no Edit or Write tool. The worst
outcome is a wrong report, which you read and discard. Runs with `--permission-mode dontAsk`,
so anything not pre-approved in `settings.json` is denied rather than hanging.

Good in a cron job, or wired into CI so every PR gets a review before you look at it.

For PR review in GitHub Actions, run `/install-github-app` inside Claude Code rather than
hand-writing the workflow. It sets up the app, the secret, and a correct workflow file for
the current action version.

## What still needs a human

Not for learning reasons. For blast-radius reasons: these are the decisions that are
expensive to reverse after code is built on top of them.

- **GP2 storage.** Now encoded as a decision rule the agent applies from the repo rather
  than a question it asks. But read the branch it took in the phase report. If it picked
  SQLite on an ephemeral filesystem, GP3 ships dead links and you find out in production.
- **GP4 classifier vs heuristic.** Decided in advance: extend the heuristic, record the
  classifier as a rejected option. Revisit only with measurements.
- **Anything that changes the FastAPI contract.** The frontend and backend deploy
  separately, so a contract change that ships out of order breaks production even though
  both sides are individually correct.
- **The smoke check.** The Stop hook covers format and types. It cannot tell you the app
  actually works. Run it.

Everything else can run unattended, with the gate armed.

## Running a phase unattended

```bash
./scripts/run-phase.sh gp-split --auto
```

`acceptEdits` with the Stop hook armed. The phase cannot end while `prettier --check` or
`tsc --noEmit` is red, so the failure mode is a stuck loop rather than a silently broken
commit. Check `git log` and the phase report afterwards, and run the app.

Chain phases only where they are independent. GP1 to GP3 to GP5 is a dependency chain,
so an unattended run of the chain compounds any mistake in GP1 through three more phases
before you look at it. Run them one at a time, `--auto` if you like, but review between.
