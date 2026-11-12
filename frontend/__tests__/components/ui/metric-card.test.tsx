import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { MetricCard, MetricGroup } from "../../../components/ui/metric-card";

describe("MetricCard", () => {
  it("associates the label (term) and value (definition) in one description list", () => {
    render(<MetricCard label="Key" value="A minor" />);
    const term = screen.getByRole("term");
    const definition = screen.getByRole("definition");
    expect(term).toHaveTextContent("Key");
    expect(definition).toHaveTextContent("A minor");
    expect(term.closest("dl")).not.toBeNull();
    expect(term.closest("dl")).toBe(definition.closest("dl"));
  });

  it("renders the unit and the secondary line inside the value pair", () => {
    render(<MetricCard label="Tempo" value={96} unit="BPM" detail="range 92-100" />);
    const definition = screen.getByRole("definition");
    expect(definition).toHaveTextContent("96");
    expect(screen.getByText("BPM")).toBeInTheDocument();
    expect(screen.getByText("range 92-100")).toBeInTheDocument();
  });

  it("is static content: nothing in it is focusable", async () => {
    const user = userEvent.setup();
    const { container } = render(<MetricCard label="Notes" value={142} />);
    await user.tab();
    expect(container.contains(document.activeElement)).toBe(false);
    expect(container.querySelector("[tabindex]")).toBeNull();
  });

  it("loading: keeps the label, shows no value, sets aria-busy", () => {
    const { container } = render(<MetricCard label="Tempo" status="loading" unit="BPM" />);
    expect(screen.getByRole("term")).toHaveTextContent("Tempo");
    expect(screen.queryByText("BPM")).toBeNull();
    expect(container.querySelector("dl")).toHaveAttribute("aria-busy", "true");
    expect(container.querySelector("[data-shape]")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByRole("definition")).toHaveTextContent("Loading");
  });

  it("ready is not busy", () => {
    const { container } = render(<MetricCard label="Key" value="C" />);
    expect(container.querySelector("dl")).not.toHaveAttribute("aria-busy");
  });

  it("unavailable: readable text, a reason, never an empty box", () => {
    render(<MetricCard label="Key" status="unavailable" reason="Too few notes in this window" />);
    expect(screen.getByRole("definition")).toHaveTextContent("Not available");
    expect(screen.getByText("Too few notes in this window")).toBeInTheDocument();
    expect(
      screen.getByRole("definition").querySelector("[data-metric-unavailable]"),
    ).not.toBeNull();
  });

  it("unavailable without a reason still says so", () => {
    render(<MetricCard label="Key" status="unavailable" />);
    expect(screen.getByRole("definition")).toHaveTextContent("Not available");
  });

  it("exposes tone as data-tone", () => {
    const { container } = render(<MetricCard label="Confidence" value="82%" tone="success" />);
    expect(container.querySelector("dl")).toHaveAttribute("data-tone", "success");
  });

  it("rejects contradictory props at compile time", () => {
    // These only need to type-check; rendering them would be meaningless.
    const never = () => (
      <>
        {/* @ts-expect-error loading has no value */}
        <MetricCard label="Key" status="loading" value="A" />
        {/* @ts-expect-error unavailable has no value */}
        <MetricCard label="Key" status="unavailable" value="A" />
        {/* @ts-expect-error ready needs a value */}
        <MetricCard label="Key" />
        {/* @ts-expect-error reason only belongs to unavailable */}
        <MetricCard label="Key" value="A" reason="x" />
        {/* @ts-expect-error label is required */}
        <MetricCard value="A" />
      </>
    );
    expect(typeof never).toBe("function");
  });
});

describe("MetricGroup", () => {
  it("is a named group of metrics", () => {
    render(
      <MetricGroup label="Analysis">
        <MetricCard label="Key" value="A minor" />
        <MetricCard label="Tempo" value={96} unit="BPM" />
      </MetricGroup>,
    );
    const group = screen.getByRole("group", { name: "Analysis" });
    expect(group.querySelectorAll("dt")).toHaveLength(2);
  });
});
