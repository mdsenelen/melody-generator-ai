/**
 * EmptyState: "there is nothing here yet, and this is what to do". Conventions in ./variants.
 *
 * Usage
 *   <EmptyState title="Load audio" body="Upload or record a clip to begin." action={<Button>Upload audio</Button>} />
 *   <EmptyState size="compact" headingLevel={3} title="No chords yet" icon={null} />
 *
 * Name        `title` is a required string, rendered as a real heading. `headingLevel` (1-6,
 *             default 2) lets the caller keep the page outline correct; the visual size does not
 *             change with the level.
 * Icon        `icon` slot; omitted renders the default mark, `icon={null}` renders none. The slot
 *             is always wrapped `aria-hidden` (decorative; the title carries the meaning), and
 *             the default mark paints with currentColor from a token, no hex.
 * Action      Any node, usually a Button. Its keyboard behaviour and focus ring are its own.
 * Size        default (page-level, generous padding) | compact (panels). Exposed as `data-size`.
 * Copy        Write title and body as directions ("Load audio to begin"), not apologies.
 *
 * States
 *   default         icon, heading, body, action
 *   hover / focus-visible / active / disabled / loading / error
 *                   do not apply to EmptyState itself: it is static content. The action slot is a
 *                   real control and owns those states. A failure is an ErrorState, a pending load
 *                   is a Skeleton or Progress.
 */
import * as React from "react";
import { cva } from "class-variance-authority";

import { cn } from "./cn";
import type { Exhaustive } from "./variants";

export type EmptyStateSize = "default" | "compact";
export type EmptyStateHeadingLevel = 1 | 2 | 3 | 4 | 5 | 6;

const SIZES: Exhaustive<EmptyStateSize> = {
  default: "gap-4 py-20",
  compact: "gap-2 py-8",
};

const root = cva("flex flex-col items-center justify-center text-center", {
  variants: { size: SIZES },
  defaultVariants: { size: "default" },
});

export function EmptyStateMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 48 48"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      className={cn("size-full", className)}
    >
      <circle cx="24" cy="24" r="22" />
      <path d="M12 24h24M24 12v24" />
    </svg>
  );
}

export type EmptyStateProps = {
  title: string;
  body?: React.ReactNode;
  action?: React.ReactNode;
  /** Decorative mark. Omit for the default; `null` for none. */
  icon?: React.ReactNode;
  headingLevel?: EmptyStateHeadingLevel;
  size?: EmptyStateSize;
  className?: string;
};

export function EmptyState({
  title,
  body,
  action,
  icon,
  headingLevel = 2,
  size = "default",
  className,
}: EmptyStateProps) {
  const Heading = `h${headingLevel}` as const;
  const mark = icon === undefined ? <EmptyStateMark /> : icon;
  return (
    <div data-size={size} className={cn(root({ size }), className)}>
      {mark !== null ? (
        <div
          data-empty-mark=""
          aria-hidden="true"
          className={cn("text-accent opacity-10", size === "compact" ? "size-8" : "size-12")}
        >
          {mark}
        </div>
      ) : null}
      <Heading
        className={cn(
          "font-display text-text font-light",
          size === "compact" ? "text-base" : "text-lg",
        )}
      >
        {title}
      </Heading>
      {body ? <p className="text-text-subtle max-w-xs text-sm leading-relaxed">{body}</p> : null}
      {action ? <div className="pt-1">{action}</div> : null}
    </div>
  );
}
