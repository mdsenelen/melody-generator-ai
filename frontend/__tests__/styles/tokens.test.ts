/**
 * Guards the token contract: the CSS variables the Tailwind theme (CSS-first
 * `@theme` in styles/tokens.css) maps to utilities must exist in every scope
 * (dark default, opt-in light), and the text/surface pairs must keep meeting
 * WCAG AA. A renamed variable fails here instead of rendering transparent.
 */
import { readFileSync } from "node:fs";
import path from "node:path";

const STYLES = path.join(__dirname, "../../styles");
const tokensCss = readFileSync(path.join(STYLES, "tokens.css"), "utf8");
const globalsCss = readFileSync(path.join(STYLES, "globals.css"), "utf8");

/** Body of the first block whose selector text starts with `selector`. */
function blocks(css: string, selector: string): string[] {
  const out: string[] = [];
  let from = 0;
  for (;;) {
    const at = css.indexOf(selector, from);
    if (at === -1) return out;
    const open = css.indexOf("{", at);
    let depth = 1;
    let i = open + 1;
    while (depth > 0 && i < css.length) {
      if (css[i] === "{") depth += 1;
      else if (css[i] === "}") depth -= 1;
      i += 1;
    }
    out.push(css.slice(open + 1, i - 1));
    from = i;
  }
}

function declarations(body: string): Map<string, string> {
  const map = new Map<string, string>();
  const clean = body.replace(/\/\*[\s\S]*?\*\//g, "");
  for (const m of clean.matchAll(/(--[a-z0-9-]+)\s*:\s*([^;]+);/gi)) {
    map.set(m[1], m[2].trim());
  }
  return map;
}

const stripped = tokensCss.replace(/\/\*[\s\S]*?\*\//g, "");
const palette = declarations(blocks(stripped, ":root {")[0]);
const darkTheme = declarations(blocks(stripped, "@theme static")[0]);
const light = declarations(blocks(stripped, '[data-theme="light"]')[0]);
const reduced = declarations(blocks(stripped, "@media (prefers-reduced-motion: reduce)")[0]);

const dark = new Map([...palette, ...darkTheme]);
const lightScope = new Map([...dark, ...light]);

const SEMANTIC_COLORS = [
  "surface",
  "surface-raised",
  "surface-muted",
  "surface-overlay",
  "border",
  "border-strong",
  "border-control",
  "text",
  "text-muted",
  "text-subtle",
  "text-faint",
  "accent",
  "accent-foreground",
  "accent-text",
  "focus-ring",
  "selection",
  "selection-text",
  "danger",
  "danger-surface",
  "danger-border",
  "danger-foreground",
  "success",
  "success-surface",
  "success-border",
  "success-foreground",
  "warning",
  "warning-surface",
  "warning-border",
  "warning-foreground",
  "info",
].map((n) => `--color-${n}`);

// Utilities the existing pages already use. They must keep resolving.
const LEGACY_COLORS = [
  "background",
  "foreground",
  "card",
  "card-foreground",
  "primary",
  "primary-foreground",
  "secondary",
  "secondary-foreground",
  "muted",
  "muted-foreground",
  "accent",
  "accent-foreground",
  "border",
  "border-strong",
  "ring",
].map((n) => `--color-${n}`);

function resolve(scope: Map<string, string>, name: string, seen = new Set<string>()): string {
  const raw = scope.get(name);
  if (raw === undefined) throw new Error(`undefined token ${name}`);
  if (seen.has(name)) throw new Error(`cycle at ${name}`);
  seen.add(name);
  const ref = raw.match(/^var\((--[a-z0-9-]+)\)$/i);
  return ref ? resolve(scope, ref[1], seen) : raw;
}

function rgb(value: string): [number, number, number] {
  const hex = value.match(/^#([0-9a-f]{6})$/i);
  if (!hex) throw new Error(`not an opaque #rrggbb colour: ${value}`);
  const n = parseInt(hex[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function luminance(value: string): number {
  const [r, g, b] = rgb(value).map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function ratio(scope: Map<string, string>, fg: string, bg: string): number {
  const [hi, lo] = [luminance(resolve(scope, fg)), luminance(resolve(scope, bg))].sort(
    (a, b) => b - a,
  );
  return (hi + 0.05) / (lo + 0.05);
}

const c = (n: string) => `--color-${n}`;
const SURFACES = ["surface", "surface-raised"];

describe("token contract", () => {
  it("globals.css pulls in the token layer and declares no colours of its own", () => {
    expect(globalsCss).toMatch(/@import "\.\/tokens\.css"/);
    expect(globalsCss).not.toMatch(/#[0-9a-f]{3,8}\b/i);
    expect(globalsCss).not.toMatch(/rgba?\(/i);
  });

  it.each([
    ["dark", dark],
    ["light", lightScope],
  ] as const)("defines every semantic colour in the %s scope", (_name, scope) => {
    for (const token of SEMANTIC_COLORS) {
      expect(scope.has(token)).toBe(true);
      expect(() => resolve(scope, token)).not.toThrow();
    }
  });

  it("keeps every legacy utility colour defined and resolvable", () => {
    for (const token of LEGACY_COLORS) {
      expect(() => resolve(dark, token)).not.toThrow();
      expect(() => resolve(lightScope, token)).not.toThrow();
    }
  });

  it("light overrides only tokens that exist in dark (no stray names)", () => {
    for (const name of light.keys()) {
      expect(dark.has(name) || name === "color-scheme").toBe(true);
    }
  });

  it("every var() reference resolves to a declared variable", () => {
    // --font-inter & co. are injected at runtime by next/font in app/layout.tsx.
    const declared = new Set([
      ...dark.keys(),
      ...light.keys(),
      "--font-inter",
      "--font-fraunces",
      "--font-jetbrains-mono",
    ]);
    const refs = [...stripped.matchAll(/var\((--[a-z0-9-]+)/gi)].map((m) => m[1]);
    expect(refs.length).toBeGreaterThan(0);
    for (const ref of refs) expect(declared.has(ref)).toBe(true);
  });

  it("derives light from dark: same palette keys, flipped surface/text order", () => {
    const l = (n: string) => luminance(resolve(lightScope, c(n)));
    const d = (n: string) => luminance(resolve(dark, c(n)));
    expect(d("surface")).toBeLessThan(d("text"));
    expect(l("surface")).toBeGreaterThan(l("text"));
  });
});

describe("WCAG AA contrast", () => {
  const scopes = [
    ["dark", dark],
    ["light", lightScope],
  ] as const;

  describe.each(scopes)("%s", (_name, scope) => {
    it.each(
      SURFACES.flatMap((s) =>
        ["text", "text-muted", "text-subtle", "accent-text"].map((t) => [t, s]),
      ),
    )("%s on %s is >= 4.5:1", (text, surface) => {
      expect(ratio(scope, c(text), c(surface))).toBeGreaterThanOrEqual(4.5);
    });

    it.each(["text", "text-muted", "text-subtle"])(
      "%s on surface-muted/overlay is >= 4.5:1",
      (t) => {
        for (const s of ["surface-muted", "surface-overlay"]) {
          expect(ratio(scope, c(t), c(s))).toBeGreaterThanOrEqual(4.5);
        }
      },
    );

    it.each(["danger", "success", "warning"])("%s text on its tinted surface is >= 4.5:1", (k) => {
      expect(ratio(scope, c(`${k}-foreground`), c(`${k}-surface`))).toBeGreaterThanOrEqual(4.5);
    });

    it.each(["danger", "success", "warning", "info"])("%s on surfaces is >= 4.5:1", (k) => {
      for (const s of SURFACES) {
        expect(ratio(scope, c(k), c(s))).toBeGreaterThanOrEqual(4.5);
      }
    });

    it("focus ring is >= 3:1 against every surface it can sit on", () => {
      for (const s of [...SURFACES, "surface-muted", "surface-overlay"]) {
        expect(ratio(scope, c("focus-ring"), c(s))).toBeGreaterThanOrEqual(3);
      }
    });

    it("control border and accent are >= 3:1 against surfaces (UI boundaries)", () => {
      for (const s of SURFACES) {
        expect(ratio(scope, c("border-control"), c(s))).toBeGreaterThanOrEqual(3);
        expect(ratio(scope, c("accent"), c(s))).toBeGreaterThanOrEqual(3);
      }
    });

    it("accent-foreground on accent is >= 3:1 (large text / UI)", () => {
      expect(ratio(scope, c("accent-foreground"), c("accent"))).toBeGreaterThanOrEqual(3);
    });
  });

  it("dark keeps the legacy accent-foreground pairing (documented sub-AA for small text)", () => {
    // #0a0a0b on #ac20e8 is 3.89:1 today. Kept as-is so adoption is no-visual-change.
    // Anyone tightening this to 4.5:1 should do it deliberately, not by accident.
    const r = ratio(dark, c("accent-foreground"), c("accent"));
    expect(r).toBeGreaterThanOrEqual(3);
    expect(r).toBeLessThan(4.5);
  });

  it("text-faint (legacy muted-foreground) passes large-text 3:1 only", () => {
    for (const s of SURFACES) {
      const r = ratio(dark, c("text-faint"), c(s));
      expect(r).toBeGreaterThanOrEqual(3);
      expect(r).toBeLessThan(4.5);
    }
  });
});

describe("scales", () => {
  it("defines spacing, radius and type-scale tokens", () => {
    for (const t of [
      "--spacing",
      "--radius",
      "--radius-xs",
      "--radius-sm",
      "--radius-md",
      "--radius-lg",
      "--radius-2xl",
      "--text-3xs",
      "--text-2xs",
      "--text-label",
      "--font-sans",
      "--font-display",
      "--font-mono",
    ]) {
      expect(dark.has(t)).toBe(true);
    }
  });

  it("keeps --radius at the 4px every existing rounded-[var(--radius)] relies on", () => {
    expect(dark.get("--radius")).toBe("4px");
  });
});

describe("motion", () => {
  const durations = [...dark.keys()].filter((k) => k.startsWith("--duration-"));

  it("defines duration tokens", () => {
    expect(durations.length).toBeGreaterThanOrEqual(3);
    expect(dark.has("--ease-standard")).toBe(true);
  });

  it("collapses every duration under prefers-reduced-motion: reduce", () => {
    for (const d of durations) {
      expect(reduced.get(d)).toBe("0.01ms");
    }
  });
});
