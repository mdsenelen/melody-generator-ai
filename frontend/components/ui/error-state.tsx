/**
 * ErrorState: the one way to say "this failed" inside a region. Conventions in ./variants.
 *
 * Usage
 *   <ErrorState title="Transcription failed" detail={error.message}
 *     action={<Button size="sm" onClick={retry}>Retry transcription</Button>} />
 *   <ErrorState tone="warning" layout="inline" announce="off" title="Server is starting up" />
 *
 * Live region   `announce` decides how a screen reader hears it:
 *                 "assertive" (default)  role="alert". An error that appears in response to a
 *                                        user action must be heard now.
 *                 "polite"               role="status". Recoverable, non-blocking notices.
 *                 "off"                  no live role. For errors present at first paint, or
 *                                        already announced by the page's own live region (the
 *                                        generate page wraps its errors in aria-live="polite";
 *                                        a second alert there would read twice). Prefer "off"
 *                                        over removing the text.
 *               Insert the component when the error occurs; a live region that is mounted
 *               already containing its text is not reliably announced.
 * Not colour     Every tone has its own icon (circle-! for danger, triangle-! for warning) plus
 *               the title text; tone is also exposed as `data-tone` and `data-error-icon`.
 *               Text sits on the tinted surface using the `-foreground` AA pair.
 * Name          `title` is a required string. It is a `<p>`, not a heading: an error banner is
 *               part of its region's content, not an outline entry.
 * Action        Retry slot; any node, usually a Button labelled with the verb of the flow
 *               ("Retry transcription"). The control owns its keyboard, focus and loading state.
 * Layout        block (default, stacked, for a region that failed) | inline (one row, a banner
 *               above content). Exposed as `data-layout`.
 * Copy          Directions, not apologies: say what to do next.
 *
 * States
 *   default         icon + title + detail + action
 *   error           this component IS the error state; tone picks severity
 *   hover / focus-visible / active / disabled / loading
 *                   do not apply to the container (static content, never focusable). The action
 *                   slot is a real control and carries them; pass a loading Button while a retry
 *                   is in flight.
 */
import * as React from "react";

import { Card } from "./card";
import { cn } from "./cn";
import type { Exhaustive } from "./variants";

export type ErrorStateTone = "danger" | "warning";
export type ErrorStateLayout = "block" | "inline";
export type ErrorStateAnnounce = "assertive" | "polite" | "off";

const CARD_VARIANT = { danger: "error", warning: "warning" } as const;

const TITLE_TONES: Exhaustive<ErrorStateTone> = {
  danger: "text-danger-foreground",
  warning: "text-warning-foreground",
};

const ICON_TONES: Exhaustive<ErrorStateTone> = {
  danger: "text-danger",
  warning: "text-warning",
};

const LAYOUTS: Exhaustive<ErrorStateLayout> = {
  block: "flex-col items-start gap-3",
  inline: "flex-row items-start gap-3",
};

const ROLES = { assertive: "alert", polite: "status", off: undefined } as const;

function ToneIcon({ tone }: { tone: ErrorStateTone }) {
  return (
    <svg
      aria-hidden="true"
      data-error-icon={tone}
      width="14"
      height="14"
      viewBox="0 0 14 14"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={cn("mt-0.5 shrink-0", ICON_TONES[tone])}
    >
      {tone === "warning" ? (
        <>
          <path d="M7 1.5L13 12H1L7 1.5z" />
          <path d="M7 5.5v3M7 10.5v.5" />
        </>
      ) : (
        <>
          <circle cx="7" cy="7" r="6" />
          <path d="M7 3.8v3.9M7 10v.4" />
        </>
      )}
    </svg>
  );
}

export type ErrorStateProps = Omit<
  React.HTMLAttributes<HTMLDivElement>,
  "role" | "title" | "children"
> & {
  title: string;
  detail?: React.ReactNode;
  action?: React.ReactNode;
  tone?: ErrorStateTone;
  layout?: ErrorStateLayout;
  announce?: ErrorStateAnnounce;
};

export function ErrorState({
  title,
  detail,
  action,
  tone = "danger",
  layout = "block",
  announce = "assertive",
  className,
  ...rest
}: ErrorStateProps) {
  return (
    <Card
      {...rest}
      variant={CARD_VARIANT[tone]}
      role={ROLES[announce]}
      data-tone={tone}
      data-layout={layout}
      className={cn("flex", LAYOUTS[layout], className)}
    >
      <div className="flex min-w-0 flex-1 items-start gap-3">
        <ToneIcon tone={tone} />
        <div className="min-w-0">
          <p className={cn("text-xs font-medium", TITLE_TONES[tone])}>{title}</p>
          {detail ? (
            <p className="text-text-muted mt-0.5 text-xs leading-relaxed break-words">{detail}</p>
          ) : null}
        </div>
      </div>
      {action ? (
        <div className={layout === "block" ? "pl-[26px]" : "shrink-0"}>{action}</div>
      ) : null}
    </Card>
  );
}
