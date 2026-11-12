import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ErrorState } from "../../../components/ui/error-state";

describe("ErrorState", () => {
  it("shows title and detail as readable text", () => {
    render(<ErrorState title="Transcription failed" detail="The file has no audible notes." />);
    expect(screen.getByText("Transcription failed")).toBeVisible();
    expect(screen.getByText("The file has no audible notes.")).toBeVisible();
  });

  it("announces by default with role=alert", () => {
    render(<ErrorState title="Generation failed" />);
    expect(screen.getByRole("alert")).toHaveTextContent("Generation failed");
  });

  it("announce=polite is a status region", () => {
    render(<ErrorState title="Server busy" announce="polite" />);
    expect(screen.getByRole("status")).toHaveTextContent("Server busy");
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("announce=off has no live region role", () => {
    const { container } = render(<ErrorState title="Server busy" announce="off" />);
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByRole("status")).toBeNull();
    expect(container.querySelector("[aria-live]")).toBeNull();
  });

  it("exposes tone as data-tone with a distinct, hidden icon (not colour alone)", () => {
    const { container, rerender } = render(<ErrorState title="T" />);
    expect(screen.getByRole("alert")).toHaveAttribute("data-tone", "danger");
    const dangerIcon = container.querySelector("[data-error-icon]");
    expect(dangerIcon).toHaveAttribute("aria-hidden", "true");
    expect(dangerIcon).toHaveAttribute("data-error-icon", "danger");
    rerender(<ErrorState title="T" tone="warning" />);
    expect(screen.getByRole("alert")).toHaveAttribute("data-tone", "warning");
    expect(container.querySelector("[data-error-icon]")).toHaveAttribute(
      "data-error-icon",
      "warning",
    );
  });

  it("layout defaults to block and can be inline", () => {
    const { rerender } = render(<ErrorState title="T" />);
    expect(screen.getByRole("alert")).toHaveAttribute("data-layout", "block");
    rerender(<ErrorState title="T" layout="inline" />);
    expect(screen.getByRole("alert")).toHaveAttribute("data-layout", "inline");
  });

  it("the retry action is keyboard operable and fires its handler", async () => {
    const user = userEvent.setup();
    const onRetry = jest.fn();
    render(
      <ErrorState title="T" action={<button onClick={onRetry}>Retry transcription</button>} />,
    );
    await user.tab();
    expect(screen.getByRole("button", { name: "Retry transcription" })).toHaveFocus();
    await user.keyboard("{Enter}");
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("passes through data attributes and merges className", () => {
    render(<ErrorState title="T" className="mt-4" data-testid="err" />);
    expect(screen.getByTestId("err")).toHaveClass("mt-4");
  });
});
