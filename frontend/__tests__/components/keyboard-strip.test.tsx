import { render, screen } from "@testing-library/react";

import { KeyboardStrip } from "../../components/keyboard-strip";

describe("KeyboardStrip", () => {
  it("names the lit notes for screen readers", () => {
    render(<KeyboardStrip pitchClasses={[0, 4, 7]} />);
    expect(screen.getByRole("img", { name: "Keyboard showing C, E, G" })).toBeInTheDocument();
  });

  it("lights one root-position voicing, not every octave's copy", () => {
    const { container } = render(<KeyboardStrip pitchClasses={[5, 9, 0, 4]} />);
    expect(container.querySelectorAll("[data-key]")).toHaveLength(24);
    const lit = Array.from(container.querySelectorAll("[data-active='true']")).map((key) =>
      Number(key.getAttribute("data-midi")),
    );
    // Fmaj7 from the first F: F, A, then C and E in the octave above.
    expect(lit.sort((a, b) => a - b)).toEqual([5, 9, 12, 16]);
  });

  it("describes an empty keyboard", () => {
    render(<KeyboardStrip pitchClasses={[]} />);
    expect(screen.getByRole("img", { name: "Keyboard, no chord selected" })).toBeInTheDocument();
  });
});
