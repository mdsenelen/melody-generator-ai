import { render, screen } from "@testing-library/react";

import { KeyboardStrip } from "../../components/keyboard-strip";

describe("KeyboardStrip", () => {
  it("names the lit notes for screen readers", () => {
    render(<KeyboardStrip pitchClasses={[0, 4, 7]} />);
    expect(screen.getByRole("img", { name: "Keyboard showing C, E, G" })).toBeInTheDocument();
  });

  it("lights every octave's copy of each pitch class across two octaves", () => {
    const { container } = render(<KeyboardStrip pitchClasses={[9, 0, 4]} />);
    expect(container.querySelectorAll("[data-key]")).toHaveLength(24);
    expect(container.querySelectorAll("[data-active='true']")).toHaveLength(6);
  });

  it("describes an empty keyboard", () => {
    render(<KeyboardStrip pitchClasses={[]} />);
    expect(screen.getByRole("img", { name: "Keyboard, no chord selected" })).toBeInTheDocument();
  });
});
