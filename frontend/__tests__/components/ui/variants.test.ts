import { cva } from "class-variance-authority";

import { cn } from "../../../components/ui/cn";
import {
  DISABLED,
  ERROR,
  FOCUS_RING,
  FOCUS_RING_INSET,
  LOADING,
  type Exhaustive,
} from "../../../components/ui/variants";

type Tone = "neutral" | "accent";

describe("shared variant pieces", () => {
  it("focus ring is keyboard-only and drawn from the focus-ring token", () => {
    expect(FOCUS_RING).toContain("focus-visible:outline-focus-ring");
    expect(FOCUS_RING).not.toMatch(/(^| )outline-none/);
    expect(FOCUS_RING_INSET).toContain("focus-visible:outline-focus-ring");
    expect(FOCUS_RING_INSET).toContain("-outline-offset-2");
  });

  it("state strings use semantic tokens only", () => {
    for (const s of [FOCUS_RING, FOCUS_RING_INSET, DISABLED, LOADING, ERROR]) {
      expect(s).not.toMatch(/#[0-9a-f]{3,8}|\b(red|green|amber|sky|slate)-\d{2,3}\b/i);
    }
  });

  it("composes with cva and a className override resolves through cn()", () => {
    const tone: Exhaustive<Tone> = {
      neutral: "bg-surface-raised text-text",
      accent: "bg-accent text-accent-foreground",
    };
    const box = cva(["px-3", FOCUS_RING, DISABLED], {
      variants: { tone },
      defaultVariants: { tone: "neutral" },
    });

    expect(box()).toContain("bg-surface-raised");
    expect(box({ tone: "accent" })).toContain("bg-accent");
    expect(cn(box(), "px-6")).toContain("px-6");
    expect(cn(box(), "px-6")).not.toContain("px-3");
  });
});
