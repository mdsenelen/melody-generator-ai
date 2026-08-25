#!/usr/bin/env bash
# Stop hook. Refuses to let a turn end while the verification gates are red.
# Escape hatch:  touch .claude/skip-gate
# Backend tests: GATE_PYTEST=1 claude

set -uo pipefail
INPUT=$(cat)

command -v git >/dev/null 2>&1 || exit 0   # fail open, never closed

ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}"
[ -n "$ROOT" ] && cd "$ROOT" 2>/dev/null || exit 0

# macOS has no GNU `timeout`. Use it when present, gtimeout if coreutils is installed,
# otherwise run without a timeout rather than failing the command outright.
TO=""
command -v timeout  >/dev/null 2>&1 && TO="timeout"
[ -z "$TO" ] && command -v gtimeout >/dev/null 2>&1 && TO="gtimeout"
tmo() { s="$1"; shift; if [ -n "$TO" ]; then "$TO" "$s" "$@"; else "$@"; fi; }
XPRE=""; [ -n "$TO" ] && XPRE="$TO 30"

jstr()  { printf '%s' "$INPUT" | sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" | head -1; }
jtrue() { printf '%s' "$INPUT" | grep -Eq "\"$1\"[[:space:]]*:[[:space:]]*true"; }

jtrue stop_hook_active && exit 0
[ "$(jstr permission_mode)" = "plan" ] && exit 0
[ -f .claude/skip-gate ] && exit 0

CHANGED=$( { git diff --name-only HEAD 2>/dev/null; git ls-files --others --exclude-standard 2>/dev/null; } | sort -u )
[ -z "$CHANGED" ] && exit 0

TS=$(printf '%s\n' "$CHANGED" | grep -E '\.(ts|tsx|js|jsx)$' || true)
PY=$(printf '%s\n' "$CHANGED" | grep -E '\.py$'            || true)
[ -z "$TS" ] && [ -z "$PY" ] && exit 0

FAIL=""

PKG=""
for d in frontend web app .; do
  [ -x "$d/node_modules/.bin/prettier" ] && { PKG="$d"; break; }
done
if [ -z "$PKG" ]; then
  for d in frontend web app .; do
    [ -f "$d/package.json" ] && { PKG="$d"; break; }
  done
fi

if [ -n "$TS" ] && [ -n "$PKG" ]; then
  if [ "$PKG" = "." ]; then
    REL="$TS"
  else
    REL=$(printf '%s\n' "$TS" | grep "^$PKG/" | sed "s|^$PKG/||" || true)
  fi

  if [ -n "$REL" ] && [ -x "$PKG/node_modules/.bin/prettier" ]; then
    if ! printf '%s\n' "$REL" | ( cd "$PKG" && xargs $XPRE ./node_modules/.bin/prettier --check ) >/tmp/gate-fmt.txt 2>&1; then
      FAIL="${FAIL}
FORMAT: prettier --check failed. Run: cd $PKG && npm run format
$(grep -v '^Checking formatting' /tmp/gate-fmt.txt | head -12)"
    fi
  elif [ -n "$REL" ]; then
    FAIL="${FAIL}
TOOLING: prettier is not installed in $PKG. The format gate cannot run.
Fix with: cd $PKG && npm install"
  fi

  if [ -f "$PKG/tsconfig.json" ] && [ -x "$PKG/node_modules/.bin/tsc" ]; then
    if ! ( cd "$PKG" && tmo 120 ./node_modules/.bin/tsc --noEmit ) >/tmp/gate-tsc.txt 2>&1; then
      FAIL="${FAIL}
TYPECHECK: tsc --noEmit failed. Run: cd $PKG && npm run typecheck
$(grep -E 'error TS' /tmp/gate-tsc.txt | head -10)"
    fi
  fi
fi

if [ -n "$PY" ] && [ "${GATE_PYTEST:-0}" = "1" ] && command -v python3 >/dev/null 2>&1; then
  if ! (cd backend && tmo 150 python3 -m pytest -q) >/tmp/gate-pytest.txt 2>&1; then
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
