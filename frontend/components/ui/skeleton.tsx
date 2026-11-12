/**
 * Skeleton: a decorative placeholder for content that is on its way. Conventions in ./variants.
 *
 * Usage
 *   <Skeleton shape="text" className="w-24" />
 *   <Skeleton shape="circle" className="size-8" />
 *   <SkeletonGroup label="Loading analysis"><Skeleton className="h-9 w-full" /></SkeletonGroup>
 *
 * Announcement  A Skeleton says nothing: aria-hidden, no text, not focusable, no role. The
 *               loading announcement belongs to the CONTAINER: put `aria-busy` on the region and
 *               let the page's own status text (live region) announce progress. Where a region
 *               has no such text, wrap the placeholders in `SkeletonGroup`: a `role="group"` with
 *               `aria-busy="true"` named by a visually hidden label. It is deliberately NOT a
 *               live region (no role=status), so it can never double-announce next to the
 *               page's own status; it is read when the user navigates to it.
 * Sizing        className (last argument to cn()). `shape` only sets the radius and a default
 *               height; give widths and any other heights through className.
 * Motion        A slow opacity pulse (`motion-safe:`). Under prefers-reduced-motion the pulse is
 *               dropped and the fill stays: a static `border-strong` tint that is still visibly a
 *               placeholder on every surface.
 *
 * States
 *   default         tinted block, pulsing unless reduced motion is requested
 *   hover / focus-visible / active / disabled / error
 *                   do not apply: static, decorative, never interactive or focusable
 *   loading         a Skeleton IS the loading state; `aria-busy` is set by SkeletonGroup or the
 *                   consuming region, not by the block
 */
import * as React from "react";
import { cva } from "class-variance-authority";

import { cn } from "./cn";
import type { Exhaustive } from "./variants";

export type SkeletonShape = "text" | "block" | "circle";

const SHAPES: Exhaustive<SkeletonShape> = {
  text: "h-3 w-full rounded-[var(--radius)]",
  block: "h-16 w-full rounded-[var(--radius)]",
  circle: "size-8 shrink-0 rounded-full",
};

const skeleton = cva("bg-border-strong/60 motion-safe:animate-pulse", {
  variants: { shape: SHAPES },
  defaultVariants: { shape: "block" },
});

export type SkeletonProps = Omit<React.HTMLAttributes<HTMLDivElement>, "children" | "role"> & {
  shape?: SkeletonShape;
};

export const Skeleton = React.forwardRef<HTMLDivElement, SkeletonProps>(function Skeleton(
  { shape = "block", className, ...rest },
  ref,
) {
  return (
    <div
      {...rest}
      ref={ref}
      aria-hidden="true"
      data-shape={shape}
      className={cn(skeleton({ shape }), className)}
    />
  );
});

export type SkeletonGroupProps = Omit<
  React.HTMLAttributes<HTMLDivElement>,
  "role" | "aria-label"
> & {
  /** What is loading, read by assistive tech ("Loading analysis"). Required. */
  label: string;
};

export const SkeletonGroup = React.forwardRef<HTMLDivElement, SkeletonGroupProps>(
  function SkeletonGroup({ label, className, children, ...rest }, ref) {
    const labelId = React.useId();
    return (
      <div
        {...rest}
        ref={ref}
        role="group"
        aria-busy="true"
        aria-labelledby={labelId}
        className={className}
      >
        <span id={labelId} className="sr-only">
          {label}
        </span>
        {children}
      </div>
    );
  },
);
