import { createRef } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { Skeleton, SkeletonGroup } from "../../../components/ui/skeleton";

describe("Skeleton", () => {
  it("is decorative: aria-hidden, no text, not focusable", async () => {
    const user = userEvent.setup();
    const { container } = render(<Skeleton />);
    const el = container.firstElementChild as HTMLElement;
    expect(el).toHaveAttribute("aria-hidden", "true");
    expect(el).toHaveTextContent("");
    expect(el).not.toHaveAttribute("tabindex");
    await user.tab();
    expect(el).not.toHaveFocus();
  });

  it("exposes the shape and defaults to block", () => {
    const { container } = render(
      <>
        <Skeleton />
        <Skeleton shape="text" />
        <Skeleton shape="circle" />
      </>,
    );
    const shapes = Array.from(container.children).map((c) => c.getAttribute("data-shape"));
    expect(shapes).toEqual(["block", "text", "circle"]);
  });

  it("merges className after its own classes so callers can size it", () => {
    const { container } = render(<Skeleton className="h-9 w-24" />);
    const el = container.firstElementChild as HTMLElement;
    expect(el).toHaveClass("h-9", "w-24");
  });

  it("forwards a ref", () => {
    const ref = createRef<HTMLDivElement>();
    render(<Skeleton ref={ref} />);
    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });
});

describe("SkeletonGroup", () => {
  it("is a busy group named by its visually hidden label", () => {
    render(
      <SkeletonGroup label="Loading analysis">
        <Skeleton />
      </SkeletonGroup>,
    );
    const group = screen.getByRole("group", { name: "Loading analysis" });
    expect(group).toHaveAttribute("aria-busy", "true");
    expect(screen.getByText("Loading analysis")).toHaveClass("sr-only");
  });

  it("is not a live region (the page owns the status text)", () => {
    render(
      <SkeletonGroup label="Loading analysis">
        <Skeleton />
      </SkeletonGroup>,
    );
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.getByRole("group")).not.toHaveAttribute("aria-live");
  });
});
