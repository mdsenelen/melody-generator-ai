/**
 * Shared pieces and the written convention for every component built on CVA.
 *
 * Usage
 *   const tone: Exhaustive<Tone> = { neutral: "...", accent: "..." };
 *   const button = cva(["inline-flex", FOCUS_RING, DISABLED], {
 *     variants: { tone },
 *     defaultVariants: { tone: "neutral" },
 *   });
 *   type ButtonProps = VariantProps<typeof button> & { className?: string };
 *   <button className={cn(button({ tone }), className)} />
 *
 * Conventions
 *  - Variants: one axis per visual decision (tone, size). Each axis is declared as
 *    `Exhaustive<Union>` so adding a union member without a style is a type error.
 *    The union type is exported next to the component; no loose `string` variants.
 *  - Default variants: always set `defaultVariants` for every axis, so `<Button />` is valid.
 *  - Compound variants: only for combinations that cannot be said per axis (for example
 *    `{ tone: "danger", loading: true }`). If an axis alone can carry it, it is not compound.
 *  - className: it is the LAST argument to `cn()` so callers can override spacing or colour
 *    and the later utility wins. A component never reads className to decide its own variant.
 *  - Tokens only: colours come from the semantic utilities in styles/tokens.css (bg-surface,
 *    text-text-muted, border-border-control ...), never raw hex or palette steps like red-500.
 *    Durations use the motion tokens: `duration-(--duration-fast)`, so they collapse under
 *    prefers-reduced-motion.
 *  - States, and how each is expressed:
 *      default         the base classes
 *      hover           `hover:` utilities, pointer devices only
 *      focus-visible   FOCUS_RING (or FOCUS_RING_INSET inside clipped containers); keyboard only
 *      active          `active:` utilities
 *      disabled        DISABLED (native `disabled:` plus `aria-disabled:`); the element must also
 *                      carry the real `disabled` or `aria-disabled` attribute
 *      loading         LOADING keyed on `aria-busy="true"`; the component also sets `disabled`
 *                      semantics so it cannot be re-triggered, and announces status via a live region
 *      error           ERROR keyed on `aria-invalid="true"`; the message goes in an element
 *                      referenced by `aria-describedby`, never colour alone
 */
import type { VariantProps } from "class-variance-authority";

/** `Record<Union, string>`: forces a style for every member of a variant's union type. */
export type Exhaustive<Union extends string> = Record<Union, string>;

/** Props type for a cva() result, re-exposed so components import one module for conventions. */
export type VariantPropsOf<Component extends (props?: never) => string> = VariantProps<Component>;

/**
 * Keyboard focus ring. An outline (not box-shadow) so it still renders in forced-colors mode and
 * is never clipped by overflow on the control itself. Offset by 2px so it is drawn against the
 * surrounding surface, which is where the 3:1 contrast requirement is measured.
 */
export const FOCUS_RING =
  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus-ring";

/** Same ring drawn inside the element's own box, for controls in overflow-hidden containers. */
export const FOCUS_RING_INSET =
  "focus-visible:outline-2 -outline-offset-2 focus-visible:outline-focus-ring";

export const DISABLED =
  "disabled:cursor-not-allowed disabled:opacity-30 aria-disabled:cursor-not-allowed aria-disabled:opacity-30";

export const LOADING = "aria-busy:cursor-progress";

export const ERROR = "aria-invalid:border-danger-border aria-invalid:text-danger";
