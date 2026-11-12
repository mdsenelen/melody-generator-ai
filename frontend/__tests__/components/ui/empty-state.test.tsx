import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { EmptyState } from "../../../components/ui/empty-state";

describe("EmptyState", () => {
  it("renders the title as a heading, level 2 by default", () => {
    render(<EmptyState title="No audio yet" />);
    expect(screen.getByRole("heading", { level: 2, name: "No audio yet" })).toBeInTheDocument();
  });

  it("lets the caller choose the heading level", () => {
    render(<EmptyState title="No audio yet" headingLevel={4} />);
    expect(screen.getByRole("heading", { level: 4, name: "No audio yet" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { level: 2 })).toBeNull();
  });

  it("renders body and action when given, and omits them otherwise", () => {
    const { rerender } = render(<EmptyState title="T" />);
    expect(screen.queryByRole("button")).toBeNull();
    rerender(<EmptyState title="T" body="Upload a file." action={<button>Upload audio</button>} />);
    expect(screen.getByText("Upload a file.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upload audio" })).toBeInTheDocument();
  });

  it("the action is reachable and operable by keyboard", async () => {
    const user = userEvent.setup();
    const onClick = jest.fn();
    render(<EmptyState title="T" action={<button onClick={onClick}>Upload audio</button>} />);
    await user.tab();
    expect(screen.getByRole("button", { name: "Upload audio" })).toHaveFocus();
    await user.keyboard("{Enter}");
    await user.keyboard(" ");
    expect(onClick).toHaveBeenCalledTimes(2);
  });

  it("hides the decorative mark from assistive tech", () => {
    const { container } = render(<EmptyState title="T" />);
    const mark = container.querySelector("[data-empty-mark]");
    expect(mark).not.toBeNull();
    expect(mark).toHaveAttribute("aria-hidden", "true");
  });

  it("hides a custom icon too, and icon={null} removes the mark", () => {
    const { container, rerender } = render(
      <EmptyState title="T" icon={<svg data-testid="custom" />} />,
    );
    expect(screen.getByTestId("custom").closest("[aria-hidden='true']")).not.toBeNull();
    rerender(<EmptyState title="T" icon={null} />);
    expect(container.querySelector("[data-empty-mark]")).toBeNull();
  });

  it("exposes the size", () => {
    const { container, rerender } = render(<EmptyState title="T" />);
    expect(container.firstElementChild).toHaveAttribute("data-size", "default");
    rerender(<EmptyState title="T" size="compact" />);
    expect(container.firstElementChild).toHaveAttribute("data-size", "compact");
  });
});
