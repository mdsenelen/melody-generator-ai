---
description: GP1. Split the 559-line page.tsx into a landing page at / and an analyse page at /analyse.
disable-model-invocation: true
---

Read @docs/GUIDED-PASS.md, sections "How I want you to operate" and "Context already confirmed".

Implement GP1. Do not touch GP2 through GP5.

1. Map the seam. Identify where landing content ends and the analyse flow begins in
   `frontend/app/page.tsx`, and which state and handlers straddle the two. Report the
   split plan as a file list with line ranges, then implement it. No approval gate on
   this phase, the blast radius is one file plus routing.
2. `/` becomes the landing page: what the app does, a link to the Colab notebook, an
   about and user-guide section.
3. `/analyse` takes the upload, transcribe, and download UI unchanged. Move it, do not
   rewrite it. Behaviour must be byte-for-byte equivalent from the user's side.
4. Fix every internal nav link. Grep for hardcoded `/` hrefs before you finish.
5. Extract shared pieces into components only where both routes actually use them. Do
   not speculatively componentise.

Verify: `npx prettier --check .`, backend pytest, dev server up, both routes load, and the
full analyse flow completes from `/analyse`. Commit GP1 alone. No push, no PR.

Report at the end: files moved, links changed, anything that surprised you.
If something in the confirmed-context section is wrong, say so and stop.
