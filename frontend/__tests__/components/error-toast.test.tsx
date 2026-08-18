import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ErrorToast } from "../../components/error-toast";

describe("ErrorToast", () => {
  it("shows the message", () => {
    render(<ErrorToast message="Something went wrong" onDismiss={() => {}} />);
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
  });

  it("calls onDismiss after 5 seconds when the callback identity is stable", () => {
    jest.useFakeTimers();
    const onDismiss = jest.fn();

    const { rerender } = render(<ErrorToast message="Upload failed" onDismiss={onDismiss} />);

    // Re-rendering with the SAME callback reference (as page.tsx now does via
    // useCallback) must not push the timer back — this is the scenario that
    // was broken when the caller passed a fresh inline arrow every render.
    rerender(<ErrorToast message="Upload failed" onDismiss={onDismiss} />);
    rerender(<ErrorToast message="Upload failed" onDismiss={onDismiss} />);

    jest.advanceTimersByTime(5000);

    expect(onDismiss).toHaveBeenCalledTimes(1);
    jest.useRealTimers();
  });

  it("does not fire the timer early if a new onDismiss identity resets it (documents the bug this guards against)", () => {
    jest.useFakeTimers();
    const first = jest.fn();
    const second = jest.fn();

    const { rerender } = render(<ErrorToast message="Upload failed" onDismiss={first} />);
    jest.advanceTimersByTime(4000);
    // A fresh callback identity (the old page.tsx bug) restarts the timer.
    rerender(<ErrorToast message="Upload failed" onDismiss={second} />);
    jest.advanceTimersByTime(4000);

    expect(first).not.toHaveBeenCalled();
    expect(second).not.toHaveBeenCalled();

    jest.advanceTimersByTime(1000);
    expect(second).toHaveBeenCalledTimes(1);
    expect(first).not.toHaveBeenCalled();

    jest.useRealTimers();
  });

  it("dismisses immediately when the close button is clicked", async () => {
    const user = userEvent.setup();
    const onDismiss = jest.fn();

    render(<ErrorToast message="Upload failed" onDismiss={onDismiss} />);
    await user.click(screen.getByRole("button"));

    expect(onDismiss).toHaveBeenCalledTimes(1);
  });
});
