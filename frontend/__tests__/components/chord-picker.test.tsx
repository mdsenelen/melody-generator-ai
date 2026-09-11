import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ChordPicker } from "../../components/chord-picker";

describe("ChordPicker", () => {
  it("shows a placeholder, then emits the formatted chord name once a root and quality are chosen", async () => {
    const user = userEvent.setup();
    const onChange = jest.fn();
    render(<ChordPicker value={null} onChange={onChange} placeholder="Chord 1" />);

    const trigger = screen.getByRole("button", { name: "Chord 1" });
    await user.click(trigger);

    await user.click(screen.getByRole("button", { name: "C#", exact: true }));
    await user.click(screen.getByRole("button", { name: "Minor 7th" }));

    expect(onChange).toHaveBeenCalledWith("C#m7");
  });

  it("shows the current chord name and offers to clear it", async () => {
    const user = userEvent.setup();
    const onChange = jest.fn();
    render(<ChordPicker value="Am" onChange={onChange} placeholder="Chord 1" />);

    expect(screen.getByRole("button", { name: /^Am/ })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear chord" }));
    expect(onChange).toHaveBeenCalledWith(null);
  });
});
