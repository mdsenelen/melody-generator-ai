import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { Pills } from "../../../components/ui/pills";

describe("Pills", () => {
  it("marks the active option pressed and calls onChange for another", async () => {
    const user = userEvent.setup();
    const onChange = jest.fn();
    render(<Pills options={["Happy", "Neutral", "Sad"]} value="Neutral" onChange={onChange} />);

    expect(screen.getByRole("button", { name: "Neutral" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Happy" })).toHaveAttribute("aria-pressed", "false");

    await user.click(screen.getByRole("button", { name: "Happy" }));
    expect(onChange).toHaveBeenCalledWith("Happy");
  });
});
