import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import ChooseProgressionPage from "../../app/choose-progression/page";

describe("ChooseProgressionPage (Compose)", () => {
  it("starts with an empty readout and an unlit keyboard", () => {
    render(<ChooseProgressionPage />);
    expect(
      screen.getByRole("heading", { level: 1, name: /build your progression/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Keyboard, no chord selected" })).toBeInTheDocument();
    expect(screen.getByText("0/4 selected")).toBeInTheDocument();
  });

  it("shows the chord you just picked in the readout and on the keyboard", async () => {
    const user = userEvent.setup();
    render(<ChooseProgressionPage />);

    await user.click(screen.getByRole("button", { name: "Chord 2" }));
    await user.click(screen.getByRole("button", { name: "C#", exact: true }));
    await user.click(screen.getByRole("button", { name: "Minor 7th" }));

    expect(screen.getByTestId("chord-readout")).toHaveTextContent("C#m7");
    expect(screen.getByRole("img", { name: "Keyboard showing C#, E, G#, B" })).toBeInTheDocument();
    expect(screen.getByText("1/4 selected")).toBeInTheDocument();
  });
});
