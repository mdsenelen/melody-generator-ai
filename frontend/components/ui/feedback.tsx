import type { ReactNode } from "react";

import { Card } from "./card";
import { Label } from "./text";

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
        <div className="flex items-center justify-between">
          <Label>{label}</Label>
          <span className="text-primary font-mono text-[10px]">{Math.round(value)}%</span>
        </div>
      ) : null}
      <div className="bg-border relative h-px overflow-hidden">
        <div
          className="bg-primary absolute top-0 left-0 h-full transition-all duration-300"
          style={{ width: `${Math.min(Math.max(value, 0), 100)}%` }}
        />
      </div>
      {sublabel ? (
        <p className="text-muted-foreground text-center font-mono text-[10px]">{sublabel}</p>
      ) : null}
    </div>
  );
}

export function EmptyState({
  title,
  body,
  action,
}: {
  title: string;
  body?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-20 text-center">
      <div className="h-12 w-12 opacity-10" aria-hidden="true">
        <svg viewBox="0 0 48 48" fill="none">
          <circle cx="24" cy="24" r="22" stroke="#AC20E8" strokeWidth="1.5" />
          <path d="M12 24h24M24 12v24" stroke="#AC20E8" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </div>
      <p className="font-display text-foreground text-lg font-light">{title}</p>
      {body ? (
        <p className="text-muted-foreground max-w-xs text-sm leading-relaxed">{body}</p>
      ) : null}
      {action ? <div className="pt-1">{action}</div> : null}
    </div>
  );
}

export function WarningBanner({ title, body }: { title: string; body?: string }) {
  return (
    <Card variant="warning" className="flex items-start gap-3">
      <svg
        width="14"
        height="14"
        viewBox="0 0 14 14"
        fill="none"
        className="mt-0.5 shrink-0"
        aria-hidden="true"
      >
        <path d="M7 1.5L13 12H1L7 1.5z" stroke="#f59e0b" strokeWidth="1.2" strokeLinejoin="round" />
        <path d="M7 5.5v3M7 10.5v.5" stroke="#f59e0b" strokeWidth="1.2" strokeLinecap="round" />
      </svg>
      <div>
        <p className="text-xs font-medium text-amber-400">{title}</p>
        {body ? (
          <p className="text-secondary-foreground mt-0.5 text-xs leading-relaxed">{body}</p>
        ) : null}
      </div>
    </Card>
  );
}

export function SuccessState({ title, body }: { title: string; body?: string }) {
  return (
    <Card variant="success" className="flex items-start gap-3">
      <svg
        width="14"
        height="14"
        viewBox="0 0 14 14"
        fill="none"
        className="mt-0.5 shrink-0"
        aria-hidden="true"
      >
        <circle cx="7" cy="7" r="6" stroke="#22c55e" strokeWidth="1.2" />
        <path
          d="M4.5 7l2 2 3-3"
          stroke="#22c55e"
          strokeWidth="1.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <div>
        <p className="text-xs font-medium text-green-400">{title}</p>
        {body ? (
          <p className="text-secondary-foreground mt-0.5 text-xs leading-relaxed">{body}</p>
        ) : null}
      </div>
    </Card>
  );
}
