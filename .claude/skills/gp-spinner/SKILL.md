---
description: GP5. Audit every text-only loading state and replace them with one shared Tailwind Spinner component.
disable-model-invocation: true
---

Read @docs/GUIDED-PASS.md, section GP5. GP2 must be merged, its polling state is a call site.

Implement GP5.

1. Use the Explore subagent to find every loading state in the frontend, so the search
   output stays out of the main context. Confirmed starting points: `upload-button.tsx`,
   `generate-variants/page.tsx`, GP2's polling state. Report the full list.
2. Build one `<Spinner />`. Read the existing Tailwind config and a current component
   first so it matches the purple and dark theme rather than introducing a second visual
   language. Props: size, label. Requirements: respects `prefers-reduced-motion`, and has
   an accessible name, since a bare spinner is invisible to a screen reader.
3. Swap every call site. Delete the text-only states, do not leave both.
4. One test covering the accessible name and the reduced-motion branch.

Verify: no text-only loading state remains, `npx prettier --check .` green. Commit GP5 alone.
