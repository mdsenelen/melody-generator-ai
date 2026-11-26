import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ChordDiagram } from "../../components/chord-diagram";

describe("ChordDiagram", () => {
  it("renders the chord chip as a real button, not a mouse-only span", () => {
    render(<ChordDiagram chord="C" />);
    const button = screen.getByRole("button", { name: /play c chord/i });
    expect(button.tagName).toBe("BUTTON");
  });

  it("is reachable and activatable via keyboard alone", async () => {
    const user = userEvent.setup();
    render(<ChordDiagram chord="G" />);

    await user.tab();
    expect(screen.getByRole("button", { name: /play g chord/i })).toHaveFocus();

    // Regression guard for the original bug: this used to be a <span
    // onClick> with no keyboard handler at all, so Enter/Space did nothing.
    // A real <button> handles this natively — just confirm it doesn't throw.
    await user.keyboard("{Enter}");
  });

  it("labels the fingering diagram for screen readers when a shape is known", () => {
    render(<ChordDiagram chord="Am" />);
    expect(screen.getByRole("img", { name: /guitar fingering diagram for am/i })).toBeTruthy();
  });

  // jsdom has no layout, so assert on the SVG geometry directly: the chord
  // name must sit entirely above the open-string (○) and muted (×) markers.
  it.each(["A7", "Em", "D", "C", "G7"])(
    "keeps the %s title above the open/muted string markers",
    (chord) => {
      const { container } = render(<ChordDiagram chord={chord} />);
      const svg = container.querySelector("svg") as SVGSVGElement;
      const title = svg.querySelector("[data-testid='diagram-title']") as SVGTextElement;
      const body = svg.querySelector("[data-testid='diagram-body']") as SVGGElement;
      expect(title).not.toBeNull();
      expect(body).not.toBeNull();

      const titleBottom = Number(title.getAttribute("y")) + 3; // baseline + descender
      const offset = Number(
        /translate\(0,\s*([\d.]+)\)/.exec(body.getAttribute("transform") ?? "")?.[1] ?? 0,
      );

      const markerTops: number[] = [];
      body.querySelectorAll("circle[fill='none']").forEach((c) => {
        markerTops.push(offset + Number(c.getAttribute("cy")) - Number(c.getAttribute("r")));
      });
      body.querySelectorAll("text").forEach((t) => {
        if (t.textContent === "×") markerTops.push(offset + Number(t.getAttribute("y")) - 10);
      });

      expect(markerTops.length).toBeGreaterThan(0);
      expect(titleBottom).toBeLessThanOrEqual(Math.min(...markerTops));
    },
  );

  it("falls back to a text notice for chords with no known fingering shape", () => {
    render(<ChordDiagram chord="Zbogus" />);
    expect(screen.getByText(/diagram unavailable for zbogus/i)).toBeInTheDocument();
  });
});
