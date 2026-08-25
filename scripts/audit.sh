#!/usr/bin/env bash
# Fully unattended, and safe to be unattended, because every agent it runs is read-only.
# Writes reports to reports/ and touches no source file.
#
#   ./scripts/audit.sh              full multi-agent review
#   ./scripts/audit.sh a11y         one auditor
#
# Good in a cron job or a pre-merge check. This is the part of the roadmap that should
# run without you: it produces findings for you to read, not changes for you to trust.

set -uo pipefail
WHICH="${1:-full}"
mkdir -p reports
STAMP=$(date +%Y%m%d-%H%M)

run() {
  local name="$1" prompt="$2" out="reports/${name}-${STAMP}.md"
  echo "Running $name ..."
  # dontAsk: pre-approved tools work, anything else is denied rather than hanging.
  # The auditors have no Edit or Write tool, so the blast radius is a report file.
  claude -p "$prompt" --permission-mode dontAsk > "$out" 2>"reports/${name}-${STAMP}.err"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "  failed (exit $rc). See reports/${name}-${STAMP}.err"
    echo "  If it was a permission denial, add the tool to .claude/settings.json allow list."
  else
    echo "  -> $out"
  fi
}

case "$WHICH" in
  full)  run senior-audit "/senior-audit" ;;
  a11y)  run a11y "Use the a11y-auditor subagent on the workspace route. Report only." ;;
  sec)   run security "Use the security-reviewer subagent on the upload path and API layer. Report only." ;;
  perf)  run perf "Use the perf-analyst subagent. Measure and report only, change nothing." ;;
  base)  run baseline "Use the repo-auditor subagent. Report only." ;;
  *)     echo "usage: $0 [full|a11y|sec|perf|base]"; exit 1 ;;
esac

echo
echo "Reports in reports/. Nothing was modified."
