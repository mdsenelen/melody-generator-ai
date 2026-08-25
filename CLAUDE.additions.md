# melody-generator-ai — Working Agreement

## What this project is
A thesis-born music generation system (CVAE + PPO) being rebuilt into a production-grade
portfolio product with real users.

- Frontend: Next.js (App Router) + React + TypeScript, deployed on Vercel
- Backend: FastAPI (Python), deployed on Render
- Domain flow: user audio in -> pitch/chord analysis -> conditional melody generation -> MIDI/WAV artifact out

## Scope rules
- The frontend is the surface being upgraded. Do not refactor Python model code unless a phase says so.
- Never commit audio files, model weights, soundfonts, `.env*`, notebook outputs, or anything over 5 MB.
- Never rewrite git history, force-push, or `git add -A` without being asked.
- If a change touches the FastAPI contract, update the shared TypeScript types in the same commit.

## Non-negotiables (every task, every phase)
1. TypeScript strict. Zero `any`. Model status with discriminated unions, never with loose booleans.
2. No new behaviour without a test. Write the failing test first, then the implementation.
3. Every async operation is cancellable (AbortController) and has explicit idle / loading / error / empty states.
4. Accessibility is part of "done": keyboard path, visible focus, ARIA live region for status changes, `prefers-reduced-motion` respected.
5. No secrets in client code. Validate uploads for MIME, extension, and size on the client, and assume the server revalidates.
6. Canvas and audio analysis never block the main thread. Heavy buffer work goes to a Worker or AudioWorklet.
7. Error boundaries are scoped. A crashing audio player must not take down the workspace.

## Definition of done
- `pnpm typecheck`, `pnpm lint`, `pnpm test` pass
- Behaviour covered at the right level: unit for logic, RTL for components, Playwright only for real user journeys
- No console errors or React warnings in the browser
- One logical commit with a conventional-commits message

## Phase discipline
Two programmes: `docs/GUIDED-PASS.md` (GP1 to GP5, product fixes) runs first, then
`docs/ROADMAP.md` (Phase 0 to 15, engineering bar). One phase per session.

This is an implementation job. Build, verify, commit, report. Do not narrate diffs, do
not explain the codebase back, do not tutor. Where a phase gives a decision rule, apply
it and state which branch you took. Stop and ask only when the repo does not settle the
question and the choice is expensive to reverse.

End each phase with a short report: what changed, what you decided and why, what you
deliberately skipped. Append it to `docs/PROGRESS.md` and commit.

## House style
- Colocate tests next to source (`Foo.tsx`, `Foo.test.tsx`).
- Domain logic lives in `src/domain/`, framework-free and directly unit-testable.
- Components receive data, they do not fetch it. Data access lives in hooks.
- Name things by what the user controls ("Generate melody"), not how the system works ("Run inference").
- No barrel files that re-export half the app.

## Useful commands
```
pnpm dev              # local dev server
pnpm typecheck        # tsc --noEmit
pnpm lint             # eslint
pnpm test             # jest + RTL
pnpm test:e2e         # playwright
pnpm build            # production build
pnpm analyze          # bundle analyzer (phase 9)
```
