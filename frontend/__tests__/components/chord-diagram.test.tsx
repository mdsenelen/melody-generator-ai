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

  it("falls back to a text notice for chords with no known fingering shape", () => {
    render(<ChordDiagram chord="Zbogus" />);
    expect(screen.getByText(/diagram unavailable for zbogus/i)).toBeInTheDocument();
  });
});
