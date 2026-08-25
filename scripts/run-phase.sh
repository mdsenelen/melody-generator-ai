#!/usr/bin/env bash
# Orchestration without giving up the approval gate.
# Sets up the branch, checks the tree is clean, and launches Claude Code in plan mode
# with the phase command already loaded. You still read and approve the plan.
#
#   ./scripts/run-phase.sh gp-split                   plan mode, you approve the plan
#   ./scripts/run-phase.sh gp-split --auto            acceptEdits, no plan gate
#   ./scripts/run-phase.sh gp-notebook --worktree     parallel, isolated checkout
#
# --auto is safe to the extent the Stop hook is armed: the phase cannot end while
# prettier or tsc is red. It does not cover the smoke check, so still run the app.

set -euo pipefail
CMD="${1:-}"
MODE="${2:-}"

if [ -z "$CMD" ]; then
  echo "usage: $0 <phase-command> [--auto|--worktree]"
  echo
  echo "guided pass:  gp-split  gp-async-jobs  gp-download-page  gp-notebook  gp-spinner"
  echo "roadmap:      baseline  domain-model  async-state  ui-kit  workspace  audio-viz"
  echo "              realtime  perf  a11y  harden  ci  showcase  senior-audit"
  exit 1
fi

BRANCH="phase/${CMD}"

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is dirty. Commit or stash first, so this phase is one revertable diff."
  git status --short
  exit 1
fi

if [ "$MODE" = "--worktree" ]; then
  # Parallel-safe: its own checkout, so a session on another branch cannot collide.
  WT="../$(basename "$PWD")-${CMD}"
  git worktree add -b "$BRANCH" "$WT" 2>/dev/null || git worktree add "$WT" "$BRANCH"
  echo "Worktree: $WT"
  cd "$WT"
else
  git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
fi

echo "Branch: $BRANCH"

if [ "$MODE" = "--auto" ]; then
  echo "Mode: acceptEdits. The Stop hook is the gate, not you."
  echo
  exec claude --permission-mode acceptEdits "/$CMD"
else
  echo "Mode: plan. Read the plan before approving."
  echo
  exec claude --permission-mode plan "/$CMD"
fi
