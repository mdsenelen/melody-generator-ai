import { createRef } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { IconButton } from "../../../components/ui/icon-button";

const Icon = <svg data-testid="icon" viewBox="0 0 8 8" />;

describe("IconButton", () => {
  it("is a button named by its label, with an aria-hidden icon", () => {
    render(<IconButton label="Play" icon={Icon} />);
    const button = screen.getByRole("button", { name: "Play" });
    expect(button).toHaveAttribute("aria-label", "Play");
    expect(button).toHaveAttribute("type", "button");
    expect(screen.getByTestId("icon").closest("[aria-hidden='true']")).not.toBeNull();
  });

  it("fires onClick on click, Enter and Space", async () => {
    const user = userEvent.setup();
    const onClick = jest.fn();
    render(<IconButton label="Play" icon={Icon} onClick={onClick} />);
    await user.click(screen.getByRole("button"));
    screen.getByRole("button").focus();
    await user.keyboard("{Enter}");
    await user.keyboard(" ");
    expect(onClick).toHaveBeenCalledTimes(3);
  });

  it("disabled blocks activation", async () => {
    const user = userEvent.setup();
    const onClick = jest.fn();
    render(<IconButton label="Play" icon={Icon} disabled onClick={onClick} />);
    expect(screen.getByRole("button")).toBeDisabled();
    await user.click(screen.getByRole("button"));
    expect(onClick).not.toHaveBeenCalled();
  });

  it("loading: aria-busy, name kept, focusable, activation blocked", async () => {
    const user = userEvent.setup();
    const onClick = jest.fn();
    render(<IconButton label="Play" icon={Icon} loading onClick={onClick} />);
    const button = screen.getByRole("button", { name: "Play" });
    expect(button).toHaveAttribute("aria-busy", "true");
    await user.tab();
    expect(button).toHaveFocus();
    await user.keyboard("{Enter}");
    await user.click(button);
    expect(onClick).not.toHaveBeenCalled();
    expect(button.querySelector("[data-button-spinner]")).not.toBeNull();
  });

  it("error: aria-invalid passes through", () => {
    render(<IconButton label="Play" icon={Icon} aria-invalid="true" />);
    expect(screen.getByRole("button")).toHaveAttribute("aria-invalid", "true");
  });

  it("is square at each size and shares the button variants", () => {
    const { rerender } = render(<IconButton label="Play" icon={Icon} size="sm" />);
    expect(screen.getByRole("button").className).toContain("size-7");
    rerender(<IconButton label="Play" icon={Icon} size="md" variant="destructive" />);
    const cls = screen.getByRole("button").className;
    expect(cls).toContain("size-10");
    expect(cls).toContain("text-danger");
    expect(cls).toContain("focus-visible:outline-focus-ring");
  });

  it("merges className and forwards ref", () => {
    const ref = createRef<HTMLButtonElement>();
    render(<IconButton ref={ref} label="Play" icon={Icon} className="size-12" />);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
    const cls = ref.current?.className.split(" ") ?? [];
    expect(cls).toContain("size-12");
    expect(cls).not.toContain("size-10");
  });

  it("does not compile without an accessible name", () => {
    // Type-level: typecheck fails if either directive becomes unused.
    // @ts-expect-error label is required
    const unnamed = <IconButton icon={Icon} />;
    // @ts-expect-error aria-label alone does not satisfy the required label
    const ariaOnly = <IconButton icon={Icon} aria-label="Play" />;
    expect(unnamed).toBeTruthy();
    expect(ariaOnly).toBeTruthy();
  });
});
