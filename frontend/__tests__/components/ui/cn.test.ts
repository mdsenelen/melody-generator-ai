import { cn } from "../../../components/ui/cn";

describe("cn", () => {
  it("joins class names", () => {
    expect(cn("a", "b")).toBe("a b");
  });

  it("drops falsy values", () => {
    expect(cn("a", false, null, undefined, 0, "", "b")).toBe("a b");
  });

  it("supports conditional objects and arrays", () => {
    expect(cn("a", { b: true, c: false }, ["d", { e: true }])).toBe("a b d e");
  });

  it("lets the later padding win over the earlier one", () => {
    expect(cn("px-4 py-2", "px-2")).toBe("py-2 px-2");
    expect(cn("p-4", "p-2")).toBe("p-2");
  });

  it("lets a caller className override a variant colour", () => {
    expect(cn("bg-accent text-accent-foreground", "bg-danger")).toBe(
      "text-accent-foreground bg-danger",
    );
  });

  it("keeps non-conflicting utilities", () => {
    expect(cn("rounded border", "border-border", "p-2")).toBe("rounded border border-border p-2");
  });

  it("treats the custom type-scale sizes as font sizes, not colours", () => {
    // text-label is a size and text-text-muted is a colour: both must survive.
    expect(cn("text-label", "text-text-muted")).toBe("text-label text-text-muted");
    // ...while two sizes still conflict.
    expect(cn("text-label", "text-xs")).toBe("text-xs");
    expect(cn("text-2xs", "text-sm")).toBe("text-sm");
  });

  it("treats the custom shadow tokens as shadows", () => {
    expect(cn("shadow-sm", "shadow-overlay")).toBe("shadow-overlay");
  });

  it("returns an empty string for no input", () => {
    expect(cn()).toBe("");
  });
});
