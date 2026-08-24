---
description: Phase 11. Security and reliability hardening, upload validation, error boundaries, and safe error surfaces.
argument-hint: "[optional: upload | errors | secrets]"
disable-model-invocation: true
---

Run the `security-reviewer` subagent first, then fix what it reports.

Scope: $ARGUMENTS

Must be true when this phase closes:
1. Upload validation sniffs the MIME type rather than trusting the extension, enforces an allowlist and a size cap before reading the file, and never interpolates a filename into a path or into the DOM.
2. No secret is reachable from the client bundle. Every `NEXT_PUBLIC_` variable is reviewed and justified.
3. Every panel sits in its own error boundary with a real recovery action, not a dead end. A crashing audio player leaves the workspace usable.
4. Every caught error maps to a `DomainError` with a safe user-facing message. No stack traces, internal endpoints, or model details reach the UI.
5. `pnpm audit --audit-level=high` is clean, or every exception is documented with a reason.

Write a test for each Critical fix. Deliberately break something (throw inside the player) and prove the boundary catches it.
