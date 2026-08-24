---
description: Phase 14. Prepare the portfolio presentation, README, demo, screenshots, and the engineering-challenges section.
argument-hint: "[optional: readme | demo | screenshots]"
disable-model-invocation: true
---

Use the `adr-writer` subagent for the writing and handle the demo setup yourself.

Assume a hiring manager gives this 60 seconds.

README order:
1. One-line pitch and the live demo link, above the fold
2. A GIF of the workspace doing the actual thing: audio in, melody out
3. Stack, in one line
4. Architecture diagram (Mermaid, rendered inline on GitHub)
5. "Engineering challenges and how they were solved": exactly three, each with the problem, the constraint, the decision, and the measured outcome. Link each to its ADR.
6. Local setup, last

Demo requirements:
- Seeded sample audio so a visitor can produce a result without uploading anything
- A visible note that the backend is on a free tier and the first request wakes it, so a cold start does not read as a broken app
- The demo must not be able to burn through inference quota: rate-limit, or serve a cached result for the sample input

Do not write "leveraged" or "utilised" anywhere. Say what was built and what it cost.

Focus: $ARGUMENTS
