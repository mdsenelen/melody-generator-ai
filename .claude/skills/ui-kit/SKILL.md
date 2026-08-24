---
description: Phase 5. Build or extend the accessible component system with Tailwind, Radix, and CVA. Pass a component name to build one.
argument-hint: "[component name, or 'foundation' for tokens]"
disable-model-invocation: true
---

Use the `ui-systems-dev` subagent.

Target: $ARGUMENTS (if empty, start with `foundation`).

For `foundation`: define the token layer first. Color, spacing, radius, and type scale as CSS variables plus a Tailwind theme extension, dark surface designed first. Add `cn()` and the CVA conventions. No component work until tokens exist.

For a named component: build it as a Radix primitive wrapped with typed CVA variants, colocate `Component.test.tsx`, and include every state: default, hover, focus-visible, active, disabled, loading, error.

Each component ships with:
- Keyboard operation and a visible focus ring
- An accessible name, and a `role` that matches what it actually is
- A test covering the interaction, written first
- A short usage example in the file header comment

Do not invent a new visual language per component. Everything derives from the tokens.
