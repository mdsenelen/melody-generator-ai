import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { Toggle } from "../../../components/ui/toggle";

describe("Toggle", () => {
  it("is a button with aria-pressed reflecting state, and toggles on click, Space and Enter", async () => {
    const user = userEvent.setup();
    render(<Toggle>Loop</Toggle>);
    const toggle = screen.getByRole("button", { name: "Loop" });
    expect(toggle).toHaveAttribute("aria-pressed", "false");

    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    toggle.focus();
    await user.keyboard(" ");
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    await user.keyboard("{Enter}");
    expect(toggle).toHaveAttribute("aria-pressed", "true");
  });

  it("supports defaultPressed (uncontrolled)", () => {
    render(<Toggle defaultPressed>Metronome</Toggle>);
    expect(screen.getByRole("button", { name: "Metronome" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("supports pressed + onPressedChange (controlled)", async () => {
    const user = userEvent.setup();
    const onPressedChange = jest.fn();
    function Controlled() {
      const [pressed, setPressed] = useState(false);
      return (
        <Toggle
          pressed={pressed}
          onPressedChange={(next) => {
            onPressedChange(next);
            setPressed(next);
          }}
        >
          Mute
        </Toggle>
      );
    }
    render(<Controlled />);
    await user.click(screen.getByRole("button", { name: "Mute" }));
    expect(onPressedChange).toHaveBeenCalledWith(true);
    expect(screen.getByRole("button", { name: "Mute" })).toHaveAttribute("aria-pressed", "true");
  });

  it("does not change when a controlled parent ignores the request", async () => {
    const user = userEvent.setup();
    render(
      <Toggle pressed={false} onPressedChange={jest.fn()}>
        Mute
      </Toggle>,
    );
    await user.click(screen.getByRole("button", { name: "Mute" }));
    expect(screen.getByRole("button", { name: "Mute" })).toHaveAttribute("aria-pressed", "false");
  });

  it("blocks toggling when disabled", async () => {
    const user = userEvent.setup();
    const onPressedChange = jest.fn();
    render(
      <Toggle disabled onPressedChange={onPressedChange}>
        Loop
      </Toggle>,
    );
    const toggle = screen.getByRole("button", { name: "Loop" });
    expect(toggle).toBeDisabled();
    await user.click(toggle);
    expect(onPressedChange).not.toHaveBeenCalled();
  });

  it("when loading: aria-busy, swallows activation, stays focusable", async () => {
    const user = userEvent.setup();
    const onPressedChange = jest.fn();
    render(
      <Toggle loading onPressedChange={onPressedChange}>
        Loop
      </Toggle>,
    );
    const toggle = screen.getByRole("button", { name: "Loop" });
    expect(toggle).toHaveAttribute("aria-busy", "true");
    expect(toggle).not.toBeDisabled();
    await user.click(toggle);
    toggle.focus();
    expect(toggle).toHaveFocus();
    await user.keyboard(" {Enter}");
    expect(onPressedChange).not.toHaveBeenCalled();
    expect(toggle).toHaveAttribute("aria-pressed", "false");
  });

  it("icon-only form exposes its label and hides the icon", () => {
    render(<Toggle label="Mute track" icon={<svg data-testid="icon" />} />);
    const toggle = screen.getByRole("button", { name: "Mute track" });
    expect(screen.getByTestId("icon").closest("[aria-hidden='true']")).not.toBeNull();
    expect(toggle).toHaveAttribute("aria-pressed", "false");
  });

  it("shows a non-colour pressed indicator", () => {
    render(<Toggle defaultPressed>Loop</Toggle>);
    expect(document.querySelector("[data-toggle-indicator]")).toBeInTheDocument();
  });

  it("requires a label for an icon-only toggle", () => {
    // @ts-expect-error icon-only toggle needs a label
    render(<Toggle icon={<svg />} />);
  });
});
