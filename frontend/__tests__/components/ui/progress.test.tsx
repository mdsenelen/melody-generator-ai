import { render, screen } from "@testing-library/react";

import { Progress } from "../../../components/ui/progress";

describe("Progress", () => {
  it("is a named progressbar", () => {
    render(<Progress aria-label="Transcription" value={30} />);
    expect(screen.getByRole("progressbar", { name: "Transcription" })).toBeInTheDocument();
  });

  it("can be named by aria-labelledby", () => {
    render(
      <>
        <span id="x">Generation</span>
        <Progress aria-labelledby="x" value={30} />
      </>,
    );
    expect(screen.getByRole("progressbar", { name: "Generation" })).toBeInTheDocument();
  });

  it("renders a visible label wired as the name", () => {
    render(<Progress label="Transcribing" value={30} />);
    expect(screen.getByText("Transcribing")).toBeVisible();
    expect(screen.getByRole("progressbar", { name: "Transcribing" })).toBeInTheDocument();
  });

  it("determinate exposes now, max and text", () => {
    render(<Progress aria-label="T" value={3} max={12} />);
    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuenow", "3");
    expect(bar).toHaveAttribute("aria-valuemax", "12");
    expect(bar).toHaveAttribute("aria-valuetext", "25%");
  });

  it("defaults max to 100 and clamps out-of-range values", () => {
    const { rerender } = render(<Progress aria-label="T" value={150} />);
    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuemax", "100");
    expect(bar).toHaveAttribute("aria-valuenow", "100");
    rerender(<Progress aria-label="T" value={-5} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "0");
    rerender(<Progress aria-label="T" value={Number.NaN} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "0");
  });

  it("shows optional mono value text", () => {
    render(<Progress aria-label="T" value={42} showValue />);
    expect(screen.getByText("42%")).toBeInTheDocument();
  });

  it("indeterminate has no aria-valuenow", () => {
    render(<Progress aria-label="Warming up" indeterminate />);
    const bar = screen.getByRole("progressbar");
    expect(bar).not.toHaveAttribute("aria-valuenow");
    expect(bar).toHaveAttribute("data-state", "indeterminate");
  });

  it("is not a live region", () => {
    const { container } = render(<Progress aria-label="T" value={10} label="Visible" />);
    expect(container.querySelector("[role=status],[aria-live]")).toBeNull();
  });

  it("exposes tone as data-tone and the error tone with a glyph", () => {
    const { rerender } = render(<Progress aria-label="T" value={10} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("data-tone", "accent");
    rerender(<Progress aria-label="T" value={10} tone="danger" />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("data-tone", "danger");
    expect(document.querySelector("[data-progress-error]")).toBeInTheDocument();
    rerender(<Progress aria-label="T" value={10} tone="success" />);
    expect(document.querySelector("[data-progress-error]")).toBeNull();
  });

  it("requires an accessible name and a mode", () => {
    // @ts-expect-error unnamed progress does not compile
    render(<Progress value={10} />);
    // @ts-expect-error neither value nor indeterminate
    render(<Progress aria-label="T" />);
    // @ts-expect-error value and indeterminate together
    render(<Progress aria-label="T" value={1} indeterminate />);
  });
});
