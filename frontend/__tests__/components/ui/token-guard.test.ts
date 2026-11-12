/**
 * Source guard for the UI kit: components/ui/*.tsx must style through tokens only.
 * Fails on raw hex colours, palette steps (red-500, amber-400 ...), legacy alias utilities
 * (bg-primary, text-muted-foreground, ring-ring ...) and arbitrary pixel font sizes
 * (text-[10px]; use text-3xs / text-2xs / text-label).
 *
 * Files not yet migrated are on LEGACY_ALLOWLIST. It is exact in both directions: a file not on
 * the list must be clean, and a file on the list must still have at least one violation, so the
 * entry has to be removed the moment the file is migrated.
 */
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";

const UI_DIR = path.join(__dirname, "../../../components/ui");

const LEGACY_ALLOWLIST = [
  "badge.tsx",
  "chord-sequence.tsx",
  "pills.tsx",
  "pitch-histogram.tsx",
  "slider.tsx", // TemperatureInput only; Slider itself is on tokens
  "text.tsx",
  "variant-picker.tsx",
];

const PREFIX = "(?:bg|text|border|ring|outline|fill|stroke|from|via|to|divide|shadow|accent|caret)";
const LEGACY_ALIASES =
  "(?:primary|primary-foreground|secondary|secondary-foreground|card|card-foreground|muted|muted-foreground|background|foreground|ring)";
const PALETTES =
  "(?:red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose|slate|gray|zinc|neutral|stone)";

const RULES: Array<[string, RegExp]> = [
  ["raw hex", /#[0-9a-fA-F]{3,8}\b/],
  ["palette step", new RegExp(`(?<![\\w-])${PREFIX}-${PALETTES}-\\d{2,3}`)],
  // `bg-black/NN` is exempt: the Dialog scrim is a theme-independent dim, not a surface colour.
  ["white/black", new RegExp(`(?<![\\w-])${PREFIX}-(?:white|black)(?![\\w/-])`)],
  ["legacy alias", new RegExp(`(?<![\\w-])${PREFIX}-${LEGACY_ALIASES}(?![\\w-])`)],
  ["arbitrary px font size", /text-\[\d+px\]/],
];

function violations(source: string): string[] {
  return RULES.filter(([, re]) => re.test(source)).map(([name]) => name);
}

const files = readdirSync(UI_DIR).filter((f) => f.endsWith(".tsx"));

describe("components/ui token guard", () => {
  it("scans the directory", () => {
    expect(files.length).toBeGreaterThan(10);
    for (const f of LEGACY_ALLOWLIST) expect(files).toContain(f);
  });

  it("the rules themselves catch what they claim to", () => {
    expect(violations('className="bg-[#120808]"')).toContain("raw hex");
    expect(violations('className="text-amber-400"')).toContain("palette step");
    expect(violations('className="focus-visible:ring-ring/60"')).toContain("legacy alias");
    expect(violations('className="text-muted-foreground"')).toContain("legacy alias");
    expect(violations('className="text-[10px]"')).toContain("arbitrary px font size");
    expect(violations('className="text-text-muted bg-surface border-border-control"')).toEqual([]);
    expect(violations('className="bg-accent text-accent-text outline-focus-ring"')).toEqual([]);
  });

  it.each(files.filter((f) => !LEGACY_ALLOWLIST.includes(f)))("%s is on tokens", (file) => {
    const found = violations(readFileSync(path.join(UI_DIR, file), "utf8"));
    expect(found).toEqual([]);
  });

  it.each(LEGACY_ALLOWLIST)(
    "%s is still legacy (remove it from the allowlist once clean)",
    (file) => {
      expect(violations(readFileSync(path.join(UI_DIR, file), "utf8")).length).toBeGreaterThan(0);
    },
  );
});
