/**
 * Joins class-name fragments, skipping falsy ones. A tiny stand-in for
 * `clsx`/`tailwind-merge` -- no dependency, no class de-duplication.
 */
export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}
