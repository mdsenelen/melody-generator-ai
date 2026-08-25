#!/usr/bin/env bash
# Stop hook. Fires when Claude thinks it is done, and refuses to let the turn end
# while the verification gates are red. Claude sees the failure on stderr and has to
# fix it before it can stop.
#
# This turns "don't break what already works" from a request into a mechanical rule.
# A prompt instruction decays over a long session. A hook does not.
#
# No jq dependency: macOS does not ship it, and a hook that silently no-ops is worse
# than no hook.
#
# Escape hatch:  touch .claude/skip-gate     (delete it to re-arm)
# Backend tests: GATE_PYTEST=1 claude        (opt in, they are slow)

set -uo pipefail
INPUT=$(cat)

command -v git >/dev/null 2>&1 || exit 0   # fail open, never closed

# Anchor to the repo root. Claude Code may be launched from frontend/ or backend/, and
# every path below is root-relative. Without this the gate silently checks nothing.
ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}"
[ -n "$ROOT" ] && cd "$ROOT" 2>/dev/null || exit 0

jstr()  { printf '%s' "$INPUT" | sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" | head -1; }
jtrue() { printf '%s' "$INPUT" | grep -Eq "\"$1\"[[:space:]]*:[[:space:]]*true"; }

# 1. Never loop. If we already blocked once this turn, let Claude stop.
jtrue stop_hook_active && exit 0

# 2. Planning turns write no code, so there is nothing to verify.
[ "$(jstr permission_mode)" = "plan" ] && exit 0

# 3. Manual override.
[ -f .claude/skip-gate ] && exit 0

# 4. Only gate when something actually changed. A conversation turn is not a build.
CHANGED=$( { git diff --name-only HEAD 2>/dev/null; git ls-files --others --exclude-standard 2>/dev/null; } | sort -u )
[ -z "$CHANGED" ] && exit 0

TS=$(printf '%s\n' "$CHANGED" | grep -E '\.(ts|tsx|js|jsx)$' || true)
PY=$(printf '%s\n' "$CHANGED" | grep -E '\.py$'            || true)
[ -z "$TS" ] && [ -z "$PY" ] && exit 0

FAIL=""

# A missing tool is not a failing check. Skip what is not installed.
have() { timeout 20 npx --no-install "$1" --version >/dev/null 2>&1; }

if [ -n "$TS" ] && have prettier; then
  if ! printf '%s\n' "$TS" | xargs -r timeout 30 npx --no-install prettier --check >/tmp/gate-fmt.txt 2>&1; then
    FAIL="${FAIL}
FORMAT: prettier --check failed on:
$(head -15 /tmp/gate-fmt.txt)"
  fi
fi

if [ -n "$TS" ] && [ -f frontend/tsconfig.json ] && (cd frontend && have tsc); then
  if ! (cd frontend && timeout 90 npx --no-install tsc --noEmit) >/tmp/gate-tsc.txt 2>&1; then
    FAIL="${FAIL}
TYPECHECK: tsc --noEmit failed:
$(grep -E 'error TS' /tmp/gate-tsc.txt | head -10)"
  fi
fi

if [ -n "$PY" ] && [ "${GATE_PYTEST:-0}" = "1" ] && command -v python3 >/dev/null 2>&1; then
  if ! (cd backend && timeout 150 python3 -m pytest -q) >/tmp/gate-pytest.txt 2>&1; then
    FAIL="${FAIL}
PYTEST: backend suite failed:
$(tail -15 /tmp/gate-pytest.txt)"
  fi
fi

if [ -n "$FAIL" ]; then
  cat >&2 <<MSG
The verification gate is red, so this phase is not done.
$FAIL

Fix these before stopping. Do not disable the gate, do not delete the failing test,
and do not commit on red. If a failure is genuinely pre-existing and unrelated to your
change, say so explicitly and ask me before going any further.
MSG
  exit 2
fi

exit 0
