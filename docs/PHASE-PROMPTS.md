# Phase prompts

The slash commands in `.claude/skills/` already contain these. Use this file when you want
to paste a prompt directly, adapt one, or run a phase in a fresh session without the config
installed. Each prompt assumes `CLAUDE.md` is loaded.

Convention below: **Session** is how to start, **Prompt** is what to paste, **Done when**
is the gate. Do not move on until the gate is green.

---

## Phase 0. Baseline
Session: `claude` then `/baseline`

> Audit this repository before we change anything. Use the repo-auditor subagent so the
> command output stays out of our conversation. I want: the current typecheck, lint, test,
> and build status with error counts; every tracked file that should never have been
> committed (audio, model weights, soundfonts, logs, .env, anything over 5 MB); any
> hardcoded key or token; and the versions of React, Next, TypeScript, and the testing
> libraries. Write it to docs/BASELINE.md. Then fix only the mechanical hygiene items:
> extend .gitignore and untrack files with git rm --cached. Anything needing history
> rewriting or a credential rotation goes in a "Requires a human decision" list and you do
> not touch it. Finish by adding the npm scripts the roadmap needs if they are missing.

Done when: `docs/BASELINE.md` exists, `git status` is clean, nothing over 5 MB is tracked.

---

## Phase 1. Domain model
Session: `/clear`, then Shift+Tab for plan mode, then `/domain-model`

> Use the domain-architect subagent. Our state handling is boolean spaghetti. Model the
> real pipeline as four typed stages: AudioInput, AudioAnalysis with pitch and chords each
> carrying a confidence value, MelodyGeneration, and GeneratedArtifact for MIDI and WAV.
> Then replace every isLoading / isProcessing / isError cluster with one discriminated
> union: idle, uploading with progress, analyzing, generating with a stage, completed with
> an artifact, failed with a DomainError carrying a code, a user-facing message, and a
> retryable flag. Put all of it in src/domain/ with no React import anywhere in that
> directory. Write the transition function with an exhaustive switch so an unhandled state
> is a compile error. Show me the union and the transition table before you write anything.
> This phase changes no UI behaviour.

Done when: zero `any` in `src/domain/`, typecheck passes, the app behaves identically.

---

## Phase 2. Testing foundation
Session: `/clear`, then `/tdd <behaviour>` per item

> Use the tdd-engineer subagent, strict red-green-refactor, and show me the red failure
> every time. If a test passes on the first run, it does not test the behaviour, rewrite it.
> Cover the critical paths in this order: audio upload validation for wrong MIME, wrong
> extension, oversized file, and zero-byte file; every transition in the job state machine
> including failure and cancellation; retry after failure; abort mid-generation; and MIDI
> and WAV export producing the right filename and blob type. Query by role and accessible
> name, use user-event not fireEvent, mock the network with MSW and never mock our own
> modules, and put the Web Audio, Canvas, and File fakes in src/test/fakes/ so every test
> shares them.

Done when: every critical path above has a test, and you watched each one fail first.

---

## Phase 3. Playwright E2E
Session: `/clear`, then `/e2e all`

> Use the e2e-runner subagent. Build five journeys: the happy path from upload through
> analysis, generation, playback, and MIDI export; a rejected file for both wrong format
> and oversized; a backend 500 mid-generation followed by a successful retry; cancellation
> returning the UI to idle with nothing orphaned; and a page reload during a running job
> recovering to a sane state. Intercept the FastAPI routes and serve fixtures from
> e2e/fixtures/ so the suite never invokes inference and finishes in under two minutes.
> Select by role and accessible name only. If something cannot be selected that way, that
> is an accessibility bug, fix the component instead of adding a test id. No
> waitForTimeout, wait on conditions. When a test fails, open the page with the Playwright
> tools and look at it before you touch the test.

Done when: five journeys pass, suite under two minutes, no test ids.

---

## Phase 4. Async state
Session: `/clear`, plan mode, then `/async-state`

> Use the domain-architect subagent. Move all server state to TanStack Query and keep
> client-only state (transport position, panel layout, selected track) in React state or a
> small store. Never mirror server data into a store. Give every API call a typed contract
> in src/domain/contracts/ parsed at the boundary with a schema, no casting a fetch result.
> Thread an AbortSignal through every request so cancelling a job actually aborts it.
> Structure query keys as typed objects, not string concatenation. Make the retry policy
> explicit: back off on transient failures, never retry a validation error. Our FastAPI
> backend is on Render's free tier and cold-starts, so the first request needs a longer
> timeout and a "waking up the server" state rather than a failure. Then run tsc --noEmit
> and eliminate every remaining any. Report the before and after count.

Done when: zero `any`, cancellation aborts the real request, cold start has its own state.

---

## Phase 5. Component system
Session: `/clear`, then `/ui-kit foundation`, then `/ui-kit <component>` per component

> Use the ui-systems-dev subagent. Start with tokens, not components: color, spacing,
> radius, and type scale as CSS variables plus a Tailwind theme extension, with the dark
> surface designed first and light derived from it. Then build the primitives as Radix
> primitives wrapped in typed CVA variants: Button, IconButton, Tabs, Dialog, Tooltip,
> Slider, Toggle, Progress, MetricCard, EmptyState, ErrorState, Skeleton. Every component
> ships with default, hover, focus-visible, active, disabled, loading, and error states, a
> colocated test written first, keyboard operation, a visible focus ring, and an accessible
> name. No raw hex in any component. Pick a visual point of view derived from audio
> instrumentation and notation, and avoid the default AI-app look.

Done when: no raw hex outside the token layer, every primitive keyboard-operable and tested.

---

## Phase 6. Audio workspace
Session: `/clear`, plan mode, then `/workspace`

> Use the ui-systems-dev subagent. Build the DAW-style shell: a header with track title,
> transport, MIDI and WAV export, and settings; a left panel for audio input, source
> controls, and generation parameters; a right panel for the waveform; and a bottom row
> with the pitch and chord analysis grid beside the generated MIDI piano roll. Panels
> resize and collapse and the layout persists across reloads. On mobile it collapses to one
> stacked column with no loss of function. Density should read as a tool: tight spacing,
> monospaced numerics for time and frequency, and color carrying state and confidence
> rather than decorating. Panel content mounts lazily, the piano roll does not load until
> there is something to show, and each panel has its own error boundary. Take one
> deliberate aesthetic risk you can justify and keep everything around it quiet.

Done when: layout persists, mobile works, one panel crashing leaves the rest usable.

---

## Phase 7. Audio visualisation
Session: `/clear`, then `/audio-viz waveform` (repeat per surface)

> Use the audio-viz-dev subagent. Build the waveform first: Canvas 2D, scrubbable,
> zoomable, with the playhead synced to AudioContext.currentTime rather than a timer.
> Draw from a peaks array cached per zoom level, never from the raw AudioBuffer per frame.
> Handle devicePixelRatio, size with a ResizeObserver, drive everything from one shared
> requestAnimationFrame loop, and clean up fully on unmount. Then the pitch timeline with
> confidence shown as opacity and a hover readout of note name and frequency, then the
> piano roll with the notes also exposed to assistive tech as a list. Use one shared
> AudioContext created on a user gesture. Tell me the measured frame cost before and after.

Done when: 60 fps during playback, no leaked AudioContext, every canvas has a text alternative.

---

## Phase 8. Real-time UX
Session: `/clear`, plan mode, then `/realtime`

> Long AI jobs are the hardest UX problem in this app, and I want it built for the honest
> case: the backend cold-starts and generation takes tens of seconds. Build a stage tracker
> driven by the domain state machine rather than a fake timer, showing uploaded, pitch
> extracted, chords detected, generating, and finalising, each with what is happening and
> roughly how long it takes. Use SSE with polling as the fallback when the connection
> drops, reconnecting with backoff and no duplicate state. Wire AbortController end to end
> so cancel actually cancels. If the first request has not responded within a few seconds,
> say the server is waking up rather than showing a generic spinner. Announce stage changes
> through a throttled aria-live polite region, not every percentage tick. A job must survive
> a page reload or recover cleanly and say what happened. Test each transition first with
> the tdd-engineer subagent.

Done when: cancel aborts the real request, cold start is explained, reload recovers.

---

## Phase 9. Performance
Session: `/clear`, then `/perf`

> Use the perf-analyst subagent to measure first and do not change a line before there is a
> number. Budgets: first-load JS for the workspace route under 250 kB gzipped, no long task
> over 50 ms during playback, 60 fps on the waveform while a generation job runs. Then fix
> in this order with the audio-viz-dev subagent: move buffer analysis (peaks, RMS, pitch
> histogram) into a Worker with transferable buffers, or an AudioWorklet if it is in the
> render path; code-split the piano roll, the export pipeline, and the analysis view; remove
> render-loop waste from context providers holding fast-changing values and derived arrays
> rebuilt every frame; and trim any dependency over 50 kB gzipped that is not on the
> critical path. Report before and after for every change and revert anything that did not
> measurably improve.

Done when: all three budgets met, every change has a number attached.

---

## Phase 10. Accessibility
Session: `/clear`, then `/a11y`

> Run the a11y-auditor subagent against the workspace route, then fix the findings highest
> severity first. Specific to this app: generation status must reach a screen reader through
> a throttled aria-live polite region with failures as role alert; the waveform canvas needs
> a text alternative giving duration, detected key, tempo, and playhead position; the piano
> roll must expose its notes to assistive tech as a list or table because canvas alone is
> invisible; the transport scrub and generation parameter sliders need the full APG slider
> keyboard pattern with aria-valuetext reading as time or a note name rather than a raw
> number; confidence, state, and selection each need a second channel beyond color; and
> prefers-reduced-motion must disable the playhead animation and panel transitions. Re-run
> the auditor afterwards and show me before and after counts, with a regression test for
> each Critical fix.

Done when: zero Critical or Serious findings, the whole flow works keyboard-only.

---

## Phase 11. Security and reliability
Session: `/clear`, then `/harden`

> Run the security-reviewer subagent, then fix what it finds. When this phase closes I want
> all of these true: upload validation sniffs the MIME type rather than trusting the
> extension, enforces an allowlist and a size cap before reading the file, and never
> interpolates a filename into a path or the DOM; no secret is reachable from the client
> bundle and every NEXT_PUBLIC_ variable is justified; every panel has its own error
> boundary with a real recovery action; every caught error maps to a DomainError with a safe
> user-facing message so no stack trace, internal endpoint, or model detail reaches the UI;
> and pnpm audit at high level is clean or every exception is documented. Write a test for
> each Critical fix, and prove the boundary works by deliberately throwing inside the player.

Done when: no Critical or High findings, a thrown error in the player leaves the app usable.

---

## Phase 12. CI/CD
Session: `/clear`, then `/ci`

> Build .github/workflows/ci.yml with four jobs running in parallel on pull request and on
> push to main: quality running lint and typecheck after a frozen-lockfile install; test
> running the unit suite with coverage uploaded as an artifact; e2e installing Chromium and
> running Playwright against fixtures only, uploading the trace and HTML report on failure;
> and build running pnpm build and failing if the workspace route exceeds its 250 kB gzipped
> first-load budget. Cache the pnpm store and the Playwright browsers so a cold run stays
> under six minutes. No job may reach the real model, the real backend, or any secret. Add a
> concurrency group per branch that cancels in-progress runs. Then validate the YAML and
> dry-run each command locally so CI does not fail on something we could have caught here.
> Finish with a branch-protection note in the README listing the required checks.

Done when: CI is green on a PR, and a deliberately broken PR is actually blocked.

---

## Phase 13. ADRs
Session: `/clear`, then `/adr <decision>` once per decision

> Use the adr-writer subagent. Write an ADR for <decision>. Read the code the decision
> actually touches and pull the real numbers first: bundle size, request count, frame cost,
> response time. An ADR with no measurement and no genuinely rejected alternative is not
> worth committing. Use the context, options considered, decision, consequences format,
> write to docs/adr/NNNN-slug.md, and add a row to the index.

Decisions owed: SSE versus polling versus WebSockets; Canvas versus SVG for the waveform
and piano roll; TanStack Query versus a store; why E2E mocks inference; how the free-tier
cold start is handled in the UI.

Done when: five ADRs exist, each naming a real alternative and a real number.

---

## Phase 14. Portfolio presentation
Session: `/clear`, then `/showcase`

> Assume a hiring manager gives this 60 seconds. Rewrite the README in this order: a
> one-line pitch and the live demo link above the fold; a GIF of the workspace doing the
> actual thing, audio in and melody out; the stack in one line; a Mermaid architecture
> diagram that renders on GitHub; an "Engineering challenges and how they were solved"
> section with exactly three, each giving the problem, the constraint, the decision, and the
> measured outcome, linked to its ADR; and local setup last. For the demo: seed sample audio
> so a visitor gets a result without uploading anything, say plainly that the backend is on
> a free tier and the first request wakes it so a cold start does not read as broken, and
> rate-limit or cache the sample result so the demo cannot burn through quota. Do not write
> "leveraged" or "utilised" anywhere.

Done when: a stranger understands what you built and what it cost you, in 60 seconds.

---

## Phase 15. Senior engineering review
Session: `/clear`, then `/senior-audit`

> Run the full multi-agent audit through the release-gatekeeper subagent. Fan out in
> parallel to repo-auditor, a11y-auditor, security-reviewer, perf-analyst, and e2e-runner,
> deduplicate the findings so one root cause is one finding, and return the verdict block
> with a PASS, WARN, or FAIL per category plus a blocking and a non-blocking list. Judge
> against the bar in CLAUDE.md. Be strict: if you would not put this in front of a senior
> hiring manager, it is not a PASS.

Done when: PASS across every category, with the blocking list empty.

---

## Between every phase

```
/phase-commit <n> <name>
/clear
```

`/clear` is not optional. Carrying Phase 6's context into Phase 7 is how an agent starts
confidently editing files it half-remembers.
