import { cn } from "./cn";

/** Row of "Variant N" pills, shared by the generate-variants and result pages. */
export function VariantPicker({
  count,
  active,
  onSelect,
}: {
  count: number;
  active: number;
  onSelect: (index: number) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {Array.from({ length: count }, (_, index) => (
        <button
          key={index}
          type="button"
          onClick={() => onSelect(index)}
          aria-pressed={active === index}
          className={cn(
            "rounded-[var(--radius)] border px-4 py-2 font-mono text-xs tracking-wide uppercase transition-colors",
            active === index
              ? "border-primary bg-primary/10 text-primary"
              : "border-border text-muted-foreground hover:border-border-strong hover:text-secondary-foreground",
          )}
        >
          Variant {index + 1}
        </button>
      ))}
    </div>
  );
}
