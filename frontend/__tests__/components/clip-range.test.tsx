import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ClipRange } from "../../components/clip-range";

describe("ClipRange", () => {
  it("commits the drafted window only when the user submits", async () => {
    const onCommit = jest.fn();
    const user = userEvent.setup();
    render(<ClipRange sourceDurationSec={120} value={{ start: 0, end: 60 }} onCommit={onCommit} />);

    fireEvent.change(screen.getByLabelText(/analysis window start/i), { target: { value: "20" } });
    fireEvent.change(screen.getByLabelText(/analysis window end/i), { target: { value: "40" } });

    // dragging alone must not fire a request
    expect(onCommit).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: /analyse this section/i }));
    expect(onCommit).toHaveBeenCalledWith({ start: 20, end: 40 });
  });

  it("keeps start at least one second below end", () => {
    const onCommit = jest.fn();
    render(<ClipRange sourceDurationSec={60} value={{ start: 0, end: 30 }} onCommit={onCommit} />);

    // push start past end -> end follows, gap preserved
    fireEvent.change(screen.getByLabelText(/analysis window start/i), { target: { value: "45" } });
    const end = screen.getByLabelText(/analysis window end/i) as HTMLInputElement;
    expect(Number(end.value)).toBeGreaterThan(45);
  });

  it("disables the submit button until the window changes, and while busy", () => {
    const { rerender } = render(
      <ClipRange sourceDurationSec={60} value={{ start: 0, end: 30 }} onCommit={jest.fn()} />,
    );
    expect(screen.getByRole("button", { name: /analyse this section/i })).toBeDisabled();

    fireEvent.change(screen.getByLabelText(/analysis window end/i), { target: { value: "40" } });
    expect(screen.getByRole("button", { name: /analyse this section/i })).toBeEnabled();

    rerender(
      <ClipRange sourceDurationSec={60} value={{ start: 0, end: 30 }} onCommit={jest.fn()} busy />,
    );
    expect(screen.getByRole("button", { name: /analysing/i })).toBeDisabled();
  });

  it("announces the pending window in a live region", () => {
    render(
      <ClipRange sourceDurationSec={200} value={{ start: 0, end: 60 }} onCommit={jest.fn()} />,
    );
    fireEvent.change(screen.getByLabelText(/analysis window start/i), { target: { value: "65" } });

    // 65s -> 1:05, end follows to 1:06
    expect(screen.getByText(/pending analysis window 1:05–1:06/i)).toBeInTheDocument();
  });

  it("exposes a formatted time as aria-valuetext on each slider", () => {
    render(
      <ClipRange sourceDurationSec={200} value={{ start: 75, end: 130 }} onCommit={jest.fn()} />,
    );
    expect(screen.getByLabelText(/analysis window start/i)).toHaveAttribute(
      "aria-valuetext",
      "1:15",
    );
    expect(screen.getByLabelText(/analysis window end/i)).toHaveAttribute("aria-valuetext", "2:10");
  });
});
