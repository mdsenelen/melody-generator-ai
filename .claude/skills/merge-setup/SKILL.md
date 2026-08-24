---
description: Fold the roadmap setup files into the project's existing CLAUDE.md, .mcp.json, and settings.json without losing anything that was already there. Run once after install.sh.
disable-model-invocation: true
allowed-tools: Read Grep Glob Edit Write Bash(git status*) Bash(git diff*) Bash(ls*) Bash(cat*)
---

## What is on disk
- Existing CLAUDE.md: @CLAUDE.md
- Additions to fold in: @CLAUDE.additions.md
- Files flagged as collisions: !`ls -1 **/*.claude-setup-new .*.claude-setup-new 2>/dev/null || echo "none"`

## Task

Merge the setup files into what already exists. You are adding, not replacing.

### The rule that resolves every conflict

The existing file states **facts about this repository**. The additions state **standards
to hold the work to**. Facts win over standards every time.

- If `CLAUDE.additions.md` says `pnpm test` and the existing CLAUDE.md or `package.json`
  says `npm test`, the existing one is right. Fix the additions, not the repo.
- Same for directory paths, the Python environment, deploy targets, and env var names.
  I wrote the additions without reading the repo, so every concrete detail in them is a
  guess until you verify it against `package.json`, the actual folder structure, and the
  existing CLAUDE.md.

### Step 1: CLAUDE.md

1. Read both files fully before editing anything.
2. Categorise every line of the existing CLAUDE.md as either a fact (keep verbatim), a
   standard (may be superseded), or stale (flag, do not delete silently).
3. Produce a merged CLAUDE.md with this shape, keeping the existing content as the spine:
   - What this project is, and how to run it: **entirely from the existing file**
   - Real commands, verified against `package.json` scripts: **existing wins**
   - Repo-specific gotchas already documented (CORS, large files, soundfonts, cold starts,
     notebook quirks): **keep all of it, this is the expensive knowledge**
   - Scope rules, non-negotiables, definition of done, phase discipline: **from the additions**
   - House style: merge, and where the two disagree, follow whatever the existing code
     actually does rather than what either document claims
4. Anything you cannot reconcile goes in a short `## Open questions` section at the bottom
   rather than being dropped or guessed at.
5. Keep the result under roughly 150 lines. CLAUDE.md loads on every turn, so a bloated one
   costs context on every single request. If the merged file is longer, move reference
   material into a skill under `.claude/skills/` and link to it.

### Step 2: JSON files

For each `.claude-setup-new` file:
- `.mcp.json`: union the `mcpServers` keys. If a server name exists in both, keep the
  existing entry, it has their working credentials and transport.
- `.claude/settings.json`: union `permissions.allow` and `permissions.deny`, deduplicated.
  Keep every existing deny rule. Append the `PostToolUse` hook only if no hook already
  matches `Edit|Write`; if one does, tell me rather than chaining two formatters.
- Never drop a key you do not recognise.

### Step 3: Report and clean up

Show me a diff-style summary before deleting anything:
```
CLAUDE.md
  kept from existing:  <n> lines  (list the load-bearing ones)
  added from setup:    <n> lines
  corrected in setup:  <what I got wrong about this repo>
  flagged as stale:    <lines I did not delete but you should look at>
```

Then, and only after I confirm:
- delete `CLAUDE.additions.md` and every `*.claude-setup-new` file
- leave `CLAUDE.md` staged but uncommitted so I can read it before it lands

Do not run `git add -A` and do not commit.
