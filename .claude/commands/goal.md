---
description: Root-cause and verifiably fix the melody-generator-ai /transcribe production failures (504/502/OOM/build). Requires Vercel and Render MCP servers connected.
---

# Goal

Get `/api/transcribe` working reliably in **production** on melody-generator-ai.
This has been "fixed" multiple times and none of the fixes held, because past
attempts declared success from local tests passing rather than from checking
what's actually live. That pattern ends now that you have direct MCP access
to both Vercel and Render — you can check reality yourself. Use it.

**If the Vercel or Render MCP tools are not available**, stop and tell the
user to run `/mcp` to connect them before continuing — do not proceed on
local evidence alone when live-checking tools exist but aren't connected.

## Ground truth — do not re-derive, start from these

- Render has previously reported the service **exceeded its memory limit**
  and was auto-restarted. A past fix (permanently caching the Basic Pitch
  model in a module-level variable) may have caused or worsened this.
- Vercel has previously failed a build with `ENOENT ... next-server.js.nft.json`.
  When the build fails, frontend commits are **not live** — never treat a
  push as deployed until the MCP tools confirm the build succeeded.
- Symptom pattern: a 504 on `/api/transcribe` ("Generation timed out after
  60s"), a client retry, then a 502 the browser reports as a CORS error
  (Render's error page lacks CORS headers, so a dead/restarting instance
  shows up as "CORS" even though CORS config is not the actual problem).
- Relevant code: `backend/app/inference.py` (`_BASIC_PITCH_LOAD_LOCK`,
  `_BASIC_PITCH_MODEL`, `_run_basic_pitch_predict`, `GENERATION_TIMEOUT_SECONDS`,
  `MAX_ANALYSIS_DURATION_SEC`), `frontend/app/page.tsx` (retry-once logic),
  `frontend/app/lib/request.ts` (`requestJson`).

## Step 0 — Establish current live state (mandatory, do this first)

Before reading a single line of code, use the MCP tools to answer:

1. **Vercel**: What is the status of the most recent deployment? Did it
   build successfully? If it failed, fetch the actual build log — don't
   assume it's the same `nft.json` error as before; check.
2. **Render**: What is the current status of the melody-generator-ai web
   service? Fetch recent logs (filter for errors/restarts) and current
   memory metrics (`get_metrics`). Has it OOM'd or restarted recently? What
   does memory usage look like over the last few hours, ideally spanning a
   period with real traffic?
3. **Git**: What has actually been committed and pushed since the last two
   fix attempts? Read the real diffs — don't trust prior session summaries.

Only after you have real answers to all three should you form a hypothesis
about what's currently broken. If step 0 shows everything is actually fine
now (build green, memory stable, no recent restarts), say so plainly instead
of hunting for a problem to fix.

## Investigation approach (if Step 0 shows something is still broken)

Use subagents to parallelize:

- One subagent audits current memory footprint in `inference.py`: what's
  held in module-level state (Basic Pitch model, `_CVAE_IDDM_BUNDLE`, any
  other caches), and whether worker count means the model loads more than
  once per instance. Cross-reference its estimate against the **actual**
  Render memory metrics from Step 0, not just theoretical sizing.
- One subagent audits the Vercel build config (`next.config.js`,
  `package.json`, `.gitignore`) against the **actual** build log fetched in
  Step 0 — trace the real error to a real line, don't guess at causes for
  an error you haven't re-read fresh.
- One subagent re-checks the timeout/lock/retry chain in `inference.py` and
  `page.tsx` for regressions or half-applied changes — confirm what's
  actually in the current code, not what a past commit message claimed.

Converge on one root-cause explanation, grounded in what Step 0 actually
showed, before writing code. If the evidence says "this doesn't fit in
512MB," say that plainly — it's a valid conclusion, not a failure.

## Fix and verify — no exceptions on the verification step

1. Implement the fix.
2. Run the local test suite / build as a first-pass sanity check only — this
   is necessary but not sufficient.
3. Commit and push.
4. **Wait for and check the actual deployment via MCP tools**: poll or ask
   the user to confirm when Vercel's build for this commit completes, then
   fetch its status directly. Same for Render — confirm the new deploy is
   live and check memory metrics again after some real traffic has hit it,
   not immediately after deploy.
5. If possible, exercise the actual `/transcribe` endpoint end to end (ask
   the user to trigger it, or check Render's logs for the next real request)
   and confirm via logs that it completed without a 504/502/restart.
6. **Do not report the issue as resolved until steps 4 and 5 have produced
   real evidence from the live services.** "Tests pass and I pushed" is not
   a valid closing statement for this task.
