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
              "rounded-full border font-mono transition-colors",
              small ? "px-2.5 py-0.5 text-[11px]" : "px-3.5 py-1.5 text-[12px]",
              active
                ? "border-primary bg-primary-deep text-primary-bright"
                : "border-border-strong text-secondary-foreground hover:bg-raised hover:text-foreground",
              "focus-visible:ring-ring focus-visible:ring-offset-background focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:outline-none",
            )}
          >
            {option}
          </button>
        );
      })}
    </div>
  );
}
