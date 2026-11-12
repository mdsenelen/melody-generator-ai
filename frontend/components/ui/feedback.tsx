/**
 * Feedback components used by the pages. ProgressBar, EmptyState and WarningBanner are thin
 * adapters over the UI-kit primitives (Progress, EmptyState, ErrorState), kept so existing call
 * sites need no edit. New code should import the primitives directly. SuccessState has no
 * primitive counterpart and stays here.
 *
 * States: all static content; the primitives document theirs.
 */
import { Card } from "./card";
import { EmptyState as EmptyStatePrimitive, type EmptyStateProps } from "./empty-state";
import { ErrorState } from "./error-state";
import { Progress } from "./progress";

/** Adapter over Progress. Without a `label` it is named "Progress". */
export function ProgressBar({
  value,
  label,
  sublabel,
}: {
  value: number;
  label?: string;
  sublabel?: string;
}) {
  return (
    <div className="space-y-2">
      {label ? (
        <Progress label={label} value={value} showValue />
      ) : (
        <Progress aria-label="Progress" value={value} />
      )}
      {sublabel ? (
        <p className="text-text-subtle text-2xs text-center font-mono">{sublabel}</p>
      ) : null}
    </div>
  );
}

/** Same component as `ui/empty-state`; re-exported so the old import path keeps working. */
export function EmptyState(props: EmptyStateProps) {
  return <EmptyStatePrimitive {...props} />;
}

/**
 * Adapter over ErrorState (warning tone, inline). `announce="off"`: the old banner had no live
 * role and the pages announce it through their own aria-live wrapper.
 */
export function WarningBanner({ title, body }: { title: string; body?: string }) {
  return <ErrorState tone="warning" layout="inline" announce="off" title={title} detail={body} />;
}

export function SuccessState({ title, body }: { title: string; body?: string }) {
  return (
    <Card variant="success" className="flex items-start gap-3">
      <svg
        width="14"
        height="14"
        viewBox="0 0 14 14"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-success mt-0.5 shrink-0"
        aria-hidden="true"
      >
        <circle cx="7" cy="7" r="6" />
        <path d="M4.5 7l2 2 3-3" />
      </svg>
      <div>
        <p className="text-success text-xs font-medium">{title}</p>
        {body ? <p className="text-text-muted mt-0.5 text-xs leading-relaxed">{body}</p> : null}
      </div>
    </Card>
  );
}
