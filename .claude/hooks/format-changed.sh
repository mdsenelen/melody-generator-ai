#!/usr/bin/env bash
# Formats and lint-fixes the file Claude just edited. Never blocks the turn.
set -uo pipefail
INPUT=$(cat)
FILE=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // empty')
[ -z "$FILE" ] && exit 0
case "$FILE" in
  *.ts|*.tsx|*.js|*.jsx|*.css|*.json|*.md)
    pnpm exec prettier --write "$FILE" >/dev/null 2>&1 || true
    ;;
esac
case "$FILE" in
  *.ts|*.tsx|*.js|*.jsx)
    pnpm exec eslint --fix "$FILE" >/dev/null 2>&1 || true
    ;;
esac
exit 0
