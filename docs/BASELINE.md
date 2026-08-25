# Repository Baseline — 2026-08-25

Audit performed via `/baseline` (repo-auditor subagent) on branch `chore/claude-setup-2`, plus a hygiene pass acting on its findings. This is a point-in-time snapshot — re-run `/baseline` periodically rather than trusting this file as current forever.

## Health check

| Check | Result | Detail |
|---|---|---|
| `npm run typecheck` (frontend) | ✅ PASS | 0 errors |
| `npm run build` (frontend) | ✅ PASS | Next.js 16.3.1 (Turbopack), 1415ms, 13 static pages + 8 dynamic API routes |
| Lint | ⚠️ Not configured | No ESLint config exists. Confirmed intentional per CLAUDE.md, deferred to roadmap Phase 12 (`/ci`) — see "Open questions" below, this is unresolved, not forgotten. |
| `npm test -- --ci` (frontend) | ✅ PASS | 10 suites, 59 tests, 0 failures |
| `pytest` (backend) | ✅ PASS | 101 collected, 99 passed, 2 skipped, 4.06s collect / 5.35s run |
| E2E (Playwright) | ⚠️ Not configured | Confirmed absent, deferred to roadmap Phase 3 (`/e2e`) |

Backend test warnings: 2 librosa `n_fft too large for input signal` warnings — library behavior on short test fixtures, not project code.

## npm scripts

| Script | Before | After |
|---|---|---|
| `typecheck` | present | present |
| `test` | present | present |
| `format` / `format:check` | present | present |
| `lint` | **missing** | added — stub that prints a pointer to roadmap Phase 12 and exits 1 (ESLint itself is not installed; installing/configuring it is a separate decision, see below) |
| `test:e2e` | **missing** | added — stub that prints a pointer to roadmap Phase 3 and exits 1 (Playwright is not installed) |
| `analyze` | **missing** | added — stub that prints a pointer to install `@next/bundle-analyzer` and exits 1 |

These three stubs exist so the roadmap's slash commands and CI don't hard-fail with npm's "missing script" error; they intentionally still fail (exit 1) rather than pretend to pass, since none of the underlying tooling is installed. None of the three are wired into `.github/workflows/ci.yml` today, so this doesn't change CI's pass/fail surface.

## Large tracked files

Repo carried ~253MB of tracked binaries, almost all unreferenced by any source file. Checked with `grep -rl` across `app/`, `components/`, `hooks/`, `utils/`.

| File | Size | Referenced in source? | Action taken |
|---|---|---|---|
| `frontend/public/video5.mp4` | 35M | No | **Untracked** (`git rm --cached`, file kept on disk) |
| `frontend/public/video3.mp4` | 33M | No | **Untracked** |
| `frontend/public/video8.mp4` | 25M | No | **Untracked** |
| `frontend/public/123.mp4` | 21M | No | **Untracked** |
| `frontend/public/12328323_1920_1080_30fps.mp4` | 18M | No | **Untracked** |
| `frontend/public/background2.mp4` | 18M | No | **Untracked** |
| `frontend/public/background222.mp4` | 12M | No | **Untracked** |
| `frontend/public/12799871_1920_1080_30fps.mp4` | 11M | No | **Untracked** |
| `frontend/public/video2.mp4` | 7.9M | No | **Untracked** |
| `frontend/public/background11111.mp4` | 5.9M | No | **Untracked** |
| `frontend/public/video7.mp4` | 4.4M | No | **Untracked** |
| `frontend/public/background.mp4` | 13M | **Yes** — `app/layout.tsx:40` | Left tracked — see below |
| `backend/app/soundfonts/GeneralUser-GS.sf2` | 31M | **Yes** — `backend/app/inference.py:999-1000`, default `SOUNDFONT_PATH` | Left tracked — see below |
| `frontend/public/demo/musical-playground-demo.wav` | 284K | No | Left alone (under the 5MB large-file threshold; noted, not acted on) |

11 files (~191MB) untracked. Untracking only removes them from git's index going forward — the blobs still exist in prior commits' history, and the files remain on disk locally. This is not a history rewrite.

`.gitignore` already had `frontend/public/*.mp4`, which is why untracked files won't resurface as untracked-new; it just never affected files already committed before the pattern was added. Added `*.sf2` to `.gitignore` for the same reason (soundfont directory had no ignore coverage at all). Note added inline in `.gitignore` explaining why `background.mp4` and `GeneralUser-GS.sf2` are still tracked despite matching these patterns.

## Requires a human decision

Not acted on — listed here per the audit's own rule that anything needing history rewriting, credential rotation, or a judgment call about live assets stops here.

1. **`frontend/public/background.mp4` (13M) and `backend/app/soundfonts/GeneralUser-GS.sf2` (31M) are both live, actively-referenced assets, but still tracked in git.** Untracking them the same way as the orphaned videos would break any fresh clone or deploy that relies on git to supply them (this repo's CI/deploy pulls from git, not a separate asset store). Fixing this properly means: move both to CDN/object storage (Backblaze B2 is already live for job artifacts, per CLAUDE.md — could plausibly host these too), update `app/layout.tsx` and `SOUNDFONT_PATH` to point at the new URLs, verify the deployed app still renders/plays correctly, *then* untrack. That's a deploy-affecting change, not repo hygiene — needs a deliberate pass.
2. **Six frontend dependencies appear unused**: `express`, `body-parser`, `express-jwt`, `react-router-dom`, `ws`, `winston`, `lodash` — none of them show up in any import under `app/`, `components/`, `hooks/`, `utils/`, `lib/`, `src/` (checked individually; `moment`, `pitchy`, and `zustand` are all genuinely used and were left alone). These look like leftovers from an earlier, non-Next.js version of the frontend, possibly with a custom Express server (`react-router-dom` conflicting with Next's own router is a strong signal). Removing dependencies wasn't in this pass's authorized scope (gitignore/untrack/dead-scripts only) and risks breaking something not caught by a grep — e.g. a middleware or serverless function this audit didn't check. Worth a deliberate `npm uninstall` pass with a full build+test verification after.
3. **Repository git history still contains every untracked video's full blob content** (~191MB across prior commits). `git rm --cached` does not shrink `.git/` — the clone stays large until someone runs `git filter-repo` or equivalent, which rewrites history and requires every collaborator to re-clone. Explicitly out of scope per this audit's instructions ("never `git filter-repo`") — flagging so it's a deliberate future call, not forgotten.
4. **ESLint / Playwright / bundle-analyzer**: whether to actually install and wire these up now, versus waiting for roadmap Phases 12/3/whichever, is already flagged as an open, unresolved question in CLAUDE.md's own "Open questions" section. This baseline pass added script *stubs* only (so the commands exist and fail informatively) — it did not install any of the three tools, since that's the exact decision CLAUDE.md says is still pending.

## Dependency versions (frontend)

All within a minor/patch of latest, no urgency:

| Package | Current | Latest |
|---|---|---|
| next | 16.3.1 | 16.3.2 |
| react / react-dom | 19.1.0 | 19.2.8 |
| typescript | 6.0.3 | 7.0.2 (major) |
| jest | 30.2.0 | 30.4.2 |
| tailwindcss | 4.1.11 | 4.3.3 |

Backend (`requirements.txt`): FastAPI 0.141.1, uvicorn 0.51.0, torch 2.13.0 (CPU wheel) — all current. `pretty_midi`, `music21`, `redis`, `boto3` are unpinned (no version constraint) — worth pinning eventually for reproducible builds, not urgent.

## Secrets scan

No hardcoded API keys, tokens, or passwords found in application source. Only `.env.example` files are tracked (correct — no real `.env*` files are in git).

## Structural notes (not acted on, low priority)

- `backend/app/model/vae.py`, `utils.py`, `loss.py` are legacy (per CLAUDE.md) and unused at inference time — left alone; deletion belongs to a dedicated model-code phase per CLAUDE.md's scope rules, not this hygiene pass.
- `backend/app/soundfonts/documentation/` (CHANGELOG.html, README.html) — third-party soundfont docs, harmless, not investigated further.
- `frontend/package.json`'s `"main": "tailwind.config.js"` field is a leftover/no-op for a Next.js app (that field matters for library resolution, not applications) — cosmetic only.
