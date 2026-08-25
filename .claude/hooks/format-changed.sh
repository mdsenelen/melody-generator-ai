#!/usr/bin/env bash
# PostToolUse hook. Formats and lint-fixes the file Claude just edited.
# Never blocks the turn, and never depends on jq (macOS does not ship it).
set -uo pipefail
INPUT=$(cat)
FILE=$(printf '%s' "$INPUT" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
[ -z "$FILE" ] && exit 0
# file_path arrives absolute, but npx resolution needs the project root on hand.
ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}"
[ -n "$ROOT" ] && cd "$ROOT" 2>/dev/null || true
[ -f "$FILE" ] || exit 0
case "$FILE" in
  *.ts|*.tsx|*.js|*.jsx|*.css|*.json|*.md)
    timeout 30 npx --no-install prettier --write "$FILE" >/dev/null 2>&1 || true ;;
esac
case "$FILE" in
  *.ts|*.tsx|*.js|*.jsx)
    timeout 45 npx --no-install eslint --fix "$FILE" >/dev/null 2>&1 || true ;;
esac
exit 0
