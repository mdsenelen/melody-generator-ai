import { cn } from "./cn";

export function Pills({
  options,
  value,
  onChange,
  small = false,
  name,
}: {
  options: string[];
  value: string;
  onChange: (value: string) => void;
  small?: boolean;
  /** Accessible group name, applied as aria-label on each pill's group role via the parent's <fieldset>/<legend> in the caller. */
  name?: string;
}) {
  return (
    <div className="flex flex-wrap gap-1.5" role={name ? "group" : undefined} aria-label={name}>
      {options.map((option) => {
        const active = value === option;
        return (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            aria-pressed={active}
            className={cn(
              "rounded-[var(--radius)] border font-mono tracking-wide transition-all",
              small ? "px-2 py-0.5 text-[10px]" : "px-3 py-1 text-[11px]",
              active
                ? "border-primary bg-primary/10 text-primary"
                : "border-border text-muted-foreground hover:border-border-strong hover:text-secondary-foreground",
              "focus-visible:ring-ring/60 focus-visible:ring-offset-background focus-visible:ring-2 focus-visible:ring-offset-1 focus-visible:outline-none",
            )}
          >
            {option}
          </button>
        );
      })}
    </div>
  );
}
