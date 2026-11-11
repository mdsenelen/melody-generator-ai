import { clsx, type ClassValue } from "clsx";
import { extendTailwindMerge } from "tailwind-merge";

/**
 * tailwind-merge needs to know the custom scale names from styles/tokens.css, otherwise it reads
 * `text-label` as a text *colour* and silently drops it next to `text-text-muted`.
 * Keep these lists in step with `--text-*` and `--shadow-*` in tokens.css.
 */
const twMerge = extendTailwindMerge({
  extend: {
    classGroups: {
      "font-size": [{ text: ["3xs", "2xs", "label"] }],
      shadow: [{ shadow: ["overlay", "popover"] }],
    },
  },
});

/**
 * Joins class names (clsx: strings, arrays, `{ cls: condition }` objects, falsy dropped) and
 * resolves Tailwind conflicts so the LAST utility wins (`cn("px-4", "px-2")` -> `"px-2"`).
 * Every component passes its `className` prop last: `cn(variants({ ... }), className)`.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
