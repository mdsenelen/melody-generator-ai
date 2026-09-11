import type { ReactNode } from "react";

/** Small uppercase mono eyebrow/label used above field groups and stats. */
export function Label({ children }: { children: ReactNode }) {
  return (
    <span className="text-muted-foreground font-mono text-[10px] tracking-widest uppercase">
      {children}
    </span>
  );
}

/** Fraunces section heading with an optional supporting line. */
export function SectionHeading({ children, sub }: { children: ReactNode; sub?: string }) {
  return (
    <div className="space-y-1">
      <h2 className="font-display text-foreground text-xl font-light">{children}</h2>
      {sub ? <p className="text-muted-foreground text-sm">{sub}</p> : null}
    </div>
  );
}
