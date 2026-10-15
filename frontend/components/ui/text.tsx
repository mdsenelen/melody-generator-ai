import type { ReactNode } from "react";

/** Small uppercase mono eyebrow/label used above field groups and stats. */
export function Label({ children }: { children: ReactNode }) {
  return (
    <span className="text-muted-foreground font-mono text-[11px] font-medium tracking-[0.14em] uppercase">
      {children}
    </span>
  );
}

/** Lora section heading with an optional supporting line. */
export function SectionHeading({ children, sub }: { children: ReactNode; sub?: string }) {
  return (
    <div className="space-y-1">
      <h2 className="font-title text-foreground text-2xl">{children}</h2>
      {sub ? <p className="text-muted-foreground text-sm">{sub}</p> : null}
    </div>
  );
}

/** The page opener every route shares: eyebrow, Lora title, one-line lead. */
export function PageHeader({
  eyebrow,
  title,
  lead,
  children,
}: {
  eyebrow: string;
  title: ReactNode;
  lead?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <header className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
      <div className="max-w-2xl">
        <Label>{eyebrow}</Label>
        <h1 className="font-title text-foreground mt-3 text-[clamp(2.25rem,4vw,3rem)] leading-[1.05] font-medium tracking-[-0.015em]">
          {title}
        </h1>
        {lead ? <p className="text-secondary-foreground mt-3 text-lg leading-7">{lead}</p> : null}
      </div>
      {children ? <div className="shrink-0">{children}</div> : null}
    </header>
  );
}
