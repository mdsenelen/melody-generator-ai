# Claude Code setup for the melody-generator-ai roadmap

Everything here drops into the root of your repo. It turns the 16-phase roadmap into
slash commands, subagents, MCP servers, and a working agreement Claude Code reads on every turn.

```
install.sh                    non-destructive installer, never overwrites
CLAUDE.additions.md           rules to fold into your existing CLAUDE.md
.mcp.json                     project-scoped MCP servers
PHASE-PROMPTS.md              copy-paste prompts, one per phase
SETUP-GUIDE.md                this file, installs to docs/CLAUDE-SETUP.md
docs/ROADMAP.md               the roadmap in the form Claude reads
docs/GUIDED-PASS.md           GP1-GP5 product fixes, grounded in your codebase
docs/PROGRESS.md              phase log, appended by /phase-commit
.claude/settings.json         permissions + auto-format hook
.claude/hooks/format-changed.sh   auto-format after every edit
.claude/hooks/verify-gate.sh      Stop hook, blocks "done" while gates are red
scripts/run-phase.sh              branch setup + plan mode launch
scripts/audit.sh                  headless read-only audits
docs/AUTOMATION.md                what to automate and what not to
.claude/agents/               11 subagents
.claude/skills/               24 slash commands
```

## Already installed an earlier version?

Run the upgrade instead. It recovers a `CLAUDE.md` or `README.md` that the first version
overwrote, replaces the files that were broken, and adds the new ones. It deletes nothing.

```bash
cd ~/path/to/melody-generator-ai
bash /path/to/this/package/upgrade.sh .
git status
```

What was broken in the first version, and is fixed here:
- both hooks parsed their input with `jq`, which macOS does not ship, so they were silent
  no-ops rather than failing loudly
- both hooks assumed the working directory was the repo root, so launching `claude` from
  `frontend/` skipped the typecheck without saying so
- `CLAUDE.md` and `README.md` shipped under those names and overwrote yours

## Install (first time)

The installer never overwrites an existing file. Anything that would collide is written
alongside as `<file>.claude-setup-new` and reported, so you merge it deliberately.

```bash
cd ~/path/to/melody-generator-ai
git checkout -b chore/claude-setup          # so the whole thing is one revertable diff
bash bash /path/to/this/package/install.sh .
```

Do not `cp -r` this package into the repo root. Four files here share a name with something
you probably already have: `CLAUDE.md`, `README.md`, `.mcp.json`, and `.claude/settings.json`.
Overwriting the README is the expensive one, since that is the portfolio front door Phase 14
builds toward. The installer handles all four, and lands the setup docs in `docs/` instead.

Where things end up:

| Package file | Installed as | On collision |
|---|---|---|
| `CLAUDE.additions.md` | `CLAUDE.additions.md` | merged into your `CLAUDE.md` by `/merge-setup`, then deleted |
| `SETUP-GUIDE.md` | `docs/CLAUDE-SETUP.md` | never touches your `README.md` |
| `PHASE-PROMPTS.md` | `docs/PHASE-PROMPTS.md` | writes `.claude-setup-new` |
| `.mcp.json` | `.mcp.json` | writes `.claude-setup-new`, yours untouched |
| `.claude/settings.json` | `.claude/settings.json` | writes `.claude-setup-new`, yours untouched |
| `.claude/agents/*`, `.claude/skills/*` | same path | writes `.claude-setup-new` per file |

Then merge, in the repo, where Claude can read both sides:

```bash
claude
> /merge-setup
```

`/merge-setup` folds `CLAUDE.additions.md` into your existing `CLAUDE.md` under one rule:
your file states facts about the repository, mine states standards, and facts win. I wrote
the additions without reading your code, so every concrete detail in them is a guess. If
your project uses npm rather than pnpm, or the domain code does not live in `src/domain/`,
the merge corrects my file, not your repo. It reports what it kept, what it added, and what
I got wrong, and it will not delete anything until you confirm.

Whatever you do, keep these from your existing `CLAUDE.md`: the real run commands, the
FastAPI and Python environment setup, deploy specifics for Vercel and Render, and every
gotcha you have already paid for in debugging time (CORS config, soundfont and FluidSynth
setup, the large-file history problem, cold-start behaviour). That is the expensive
knowledge in the file. My additions are just standards, and standards are cheap to rewrite.

Finally, review and commit as one diff:

```bash
git status
git diff
git add CLAUDE.md .mcp.json .claude docs
git commit -m "chore: add Claude Code project configuration"
```

Restart Claude Code once after installing, then confirm everything loaded:

```bash
claude
/context          # subagents and skills should appear
/mcp              # server status, and OAuth for github
/doctor           # setup checkup, flags malformed frontmatter
```

Claude Code watches `.claude/agents/` and `.claude/skills/` and picks up later edits within
a few seconds. The one case needing a restart is creating a directory that did not exist
when the session started, which is exactly what the installer just did.

## Rolling it back

Everything landed on a branch and nothing was deleted, so:

```bash
git checkout -- .        # revert changes to tracked files
git clean -n             # review untracked additions first
git clean -fd            # then remove them
```

## MCP servers

`.mcp.json` is project-scoped, so it is committed and Claude Code asks you to approve it
on first use. Three servers, chosen because each one earns its context cost:

| Server | Why it is here | Phases |
|---|---|---|
| `playwright` | drives a real browser, so E2E failures get looked at instead of guessed at | 3, 6, 10 |
| `context7` | current docs for React, TanStack Query, Radix, Playwright, instead of stale training data | 4, 5, 7 |
| `github` | PRs, Actions runs, and CI failure logs without leaving the session | 12, 14 |

Equivalent CLI, if you would rather add them by hand:

```bash
claude mcp add --scope project playwright -- npx -y @playwright/mcp@latest
claude mcp add --scope project context7 -- npx -y @upstash/context7-mcp
claude mcp add --scope project --transport http github https://api.githubcopilot.com/mcp/
claude mcp list
```

Worth adding later, once the deploy story matters (check the current package name or URL
before you add these, hosted MCP endpoints move):

```bash
claude mcp add --transport http sentry https://mcp.sentry.dev/mcp   # phase 11 error monitoring
```

Keep the total at three to five servers. Every server spends context on tool definitions.

## Subagents

Eleven, in `.claude/agents/`. Five do the building, five audit, one judges.

| Agent | Model | Writes? | Phases |
|---|---|---|---|
| `repo-auditor` | haiku | no | 0, and every audit |
| `domain-architect` | opus | yes | 1, 4 |
| `tdd-engineer` | sonnet | yes | 2, and every feature after |
| `e2e-runner` | sonnet | yes | 3 |
| `ui-systems-dev` | sonnet | yes | 5, 6 |
| `audio-viz-dev` | sonnet | yes | 7, 9 |
| `a11y-auditor` | sonnet | no | 10, 15 |
| `security-reviewer` | sonnet | no | 11, 15 |
| `perf-analyst` | sonnet | no | 9, 15 |
| `adr-writer` | sonnet | yes | 13, 14 |
| `release-gatekeeper` | opus | no | 15 |

The read-only ones have `tools: Read, Grep, Glob, Bash` with no Edit or Write, so an
auditor physically cannot "helpfully" fix what it found. That separation is the point:
the agent that finds the problem is not the agent that judges whether the fix is real.

Invoke explicitly when you want a specific one: `@repo-auditor check the upload path`.
Three of them (`repo-auditor`, `domain-architect`, `audio-viz-dev`) have `memory: project`,
so they accumulate notes in `.claude/agent-memory/` across sessions. Commit that directory.

## Slash commands

One per phase, plus three utilities. All the phase commands set
`disable-model-invocation: true`, so Claude cannot decide on its own to start Phase 12.

```
/baseline        0   audit and clean the repo
/domain-model    1   types and the job state machine
/tdd <behaviour> 2   test-first implementation
/e2e [journey]   3   Playwright journeys
/async-state     4   TanStack Query, contracts, cancellation
/ui-kit [comp]   5   tokens, Radix + CVA primitives
/workspace       6   the DAW-style panel layout
/audio-viz <s>   7   waveform, pitch timeline, piano roll
/realtime        8   SSE, stages, cancellation, cold start
/perf            9   measure, then Workers and code-splitting
/a11y           10   WCAG 2.2 AA
/harden         11   upload validation, boundaries, error surface
/ci             12   GitHub Actions with real gates
/adr <decision> 13   architecture decision records
/showcase       14   README, demo, engineering-challenges section
/senior-audit   15   full multi-agent review, PASS/WARN/FAIL

/merge-setup         fold this package into your existing config (run once)
/phase-status        where you are and what is next
/phase-commit <n>    verify gates, log, commit
```

`/senior-audit` uses `context: fork` with `agent: release-gatekeeper`, so the whole audit
runs in its own context and only the verdict comes back. The nested audits it spawns never
touch your main window.

## Two tracks, and which one runs first

There are two work programmes in this package and they use overlapping numbers, so they are
labelled differently on purpose:

- **GP1 to GP5** in `docs/GUIDED-PASS.md`: product fixes to the app as it exists today.
  Grounded in verified facts about the codebase.
- **Phase 0 to 15** in `docs/ROADMAP.md`: raising the engineering bar around it.

Run the guided pass first. GP2 builds the async job flow that roadmap Phase 8 describes,
so doing them the other way around means building it twice. Roadmap Phase 0 is the one
exception worth running before anything: it is a read-only audit and it takes ten minutes.

```
/baseline          roadmap Phase 0, audit only
/gp-split          GP1
/gp-async-jobs     GP2
/gp-download-page  GP3
/gp-notebook       GP4   (independent, parallel branch is fine)
/gp-spinner        GP5
then roadmap Phase 1 onward
```

## How to actually run this

One phase per session. Between phases: `/clear`.

```
claude
> /phase-status
> /baseline
  ... review, approve the plan, let it work ...
> /phase-commit 0 baseline
> /clear
> /domain-model
```

Four things that make the difference between this working and this producing plausible slop:

1. **Plan before build on wide phases.** Press Shift+Tab to plan mode for phases 1, 4, 6, and 8. Read the plan, argue with it, then approve.
2. **Never accept a green test you did not see go red.** The `/tdd` command forces the failure to be shown. Read it.
3. **Commit at every phase boundary.** A phase that cannot be committed is a phase that is not done.
4. **Do not run phases out of order.** Phase 2 is worthless without Phase 1's state machine, and Phase 15 is theatre without Phase 12's gates.

The roadmap is ambitious. Phases 0 to 4 are the ones that change how the codebase reads;
phases 5 to 8 are the ones a hiring manager will actually see; 9 to 15 are what separates
a demo from a product. If time runs short, cut 9 and 14 before you cut 2 and 10.
