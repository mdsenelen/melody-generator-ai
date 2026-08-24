---
description: Phase 2. Implement a feature or fix strictly test-first with Jest and React Testing Library. Pass the behaviour to implement as the argument.
argument-hint: "<behaviour to implement>"
disable-model-invocation: true
---

Use the `tdd-engineer` subagent to implement the following behaviour, test-first:

$ARGUMENTS

Non-negotiable sequence, and you must show me the output of each step:
1. Write the test. Run it. Show me the red failure.
2. Write the minimum code to pass. Run it. Show me green.
3. Refactor with tests still green.

If the first run passes, the test does not actually test the behaviour. Rewrite it.

Query by role and accessible name. Use `user-event`, not `fireEvent`. Mock the network with MSW, never mock our own modules. Put Web Audio, Canvas, and File fakes in `src/test/fakes/` so every test shares them.

Finish by reporting the suite count and total runtime.
