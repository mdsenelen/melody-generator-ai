import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { Slider, TemperatureInput } from "../../../components/ui/slider";

function Harness(props: { unit?: string; step?: number; disabled?: boolean; start?: number }) {
  const [value, setValue] = useState(props.start ?? 100);
  return (
    <Slider
      label="Tempo"
      value={value}
      min={40}
      max={200}
      step={props.step}
      unit={props.unit}
      disabled={props.disabled}
      onChange={setValue}
    />
  );
}

describe("Slider", () => {
  it("exposes a named slider with value attributes and a readable value text", () => {
    render(
      <Slider label="Tempo" value={100} min={40} max={200} unit=" BPM" onChange={jest.fn()} />,
    );

    const slider = screen.getByRole("slider", { name: "Tempo" });
    expect(slider).toHaveAttribute("aria-valuemin", "40");
    expect(slider).toHaveAttribute("aria-valuemax", "200");
    expect(slider).toHaveAttribute("aria-valuenow", "100");
    expect(slider).toHaveAttribute("aria-valuetext", "100 BPM");
    expect(screen.getByText("100 BPM")).toBeInTheDocument();
  });

  it("uses the bare value as value text when there is no unit", () => {
    render(<Slider label="Variants" value={4} min={1} max={8} onChange={jest.fn()} />);
    expect(screen.getByRole("slider", { name: "Variants" })).toHaveAttribute("aria-valuetext", "4");
  });

  it("lets an explicit aria-label override the visible-label name", () => {
    render(
      <Slider
        label="Tempo"
        aria-label="Playback tempo"
        value={1}
        min={0}
        max={2}
        onChange={jest.fn()}
      />,
    );
    expect(screen.getByRole("slider", { name: "Playback tempo" })).toBeInTheDocument();
  });

  it("steps with the arrow keys and reports numbers", async () => {
    const user = userEvent.setup();
    const onChange = jest.fn();
    render(<Slider label="Tempo" value={100} min={40} max={200} step={5} onChange={onChange} />);
    screen.getByRole("slider").focus();

    await user.keyboard("{ArrowRight}");
    expect(onChange).toHaveBeenLastCalledWith(105);
    await user.keyboard("{ArrowLeft}");
    expect(onChange).toHaveBeenLastCalledWith(95);
    expect(typeof onChange.mock.calls[0][0]).toBe("number");
  });

  it("jumps to min and max with Home and End", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const slider = screen.getByRole("slider");
    slider.focus();
    await user.keyboard("{End}");
    expect(slider).toHaveAttribute("aria-valuenow", "200");
    await user.keyboard("{Home}");
    expect(slider).toHaveAttribute("aria-valuenow", "40");
  });

  it("takes a larger step with PageUp and PageDown", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const slider = screen.getByRole("slider");
    slider.focus();
    await user.keyboard("{PageUp}");
    expect(Number(slider.getAttribute("aria-valuenow"))).toBeGreaterThan(101);
    await user.keyboard("{PageDown}{PageDown}");
    expect(Number(slider.getAttribute("aria-valuenow"))).toBeLessThan(100);
  });

  it("updates the readout as the value changes", async () => {
    const user = userEvent.setup();
    render(<Harness unit=" BPM" />);
    screen.getByRole("slider").focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByText("101 BPM")).toBeInTheDocument();
  });

  it("blocks changes and is exposed as disabled", async () => {
    const user = userEvent.setup();
    const onChange = jest.fn();
    render(<Slider label="Tempo" value={100} min={40} max={200} disabled onChange={onChange} />);
    const slider = screen.getByRole("slider");
    expect(slider).toHaveAttribute("aria-disabled", "true");
    slider.focus();
    await user.keyboard("{ArrowRight}{End}");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("does not cancel its own focus ring", () => {
    // Found in a real browser: `outline-none` on the thumb sets --tw-outline-style in Tailwind v4,
    // so the focus-visible outline computed to `none`. jsdom cannot compute that, so guard the class.
    render(<Slider label="Tempo" value={90} min={60} max={160} />);
    const thumb = screen.getByRole("slider", { name: "Tempo" });
    expect(thumb.className).toContain("focus-visible:outline-2");
    expect(thumb.className.split(/\s+/)).not.toContain("outline-none");
  });

  it("marks an invalid slider with aria-invalid and a non-colour glyph", () => {
    render(
      <Slider label="Tempo" value={100} min={40} max={200} aria-invalid onChange={jest.fn()} />,
    );
    expect(screen.getByRole("slider")).toHaveAttribute("aria-invalid", "true");
    expect(document.querySelector("[data-slider-error]")).toBeInTheDocument();
  });
});

describe("TemperatureInput", () => {
  it("renders within the 0.3-2.0 range and reports changes", () => {
    const onChange = jest.fn();
    render(<TemperatureInput label="α" value={1.0} onChange={onChange} />);

    const slider = screen.getByRole("slider");
    expect(slider).toHaveAttribute("min", "0.3");
    expect(slider).toHaveAttribute("max", "2");

    fireEvent.change(slider, { target: { value: "1.5" } });
    expect(onChange).toHaveBeenCalledWith(1.5);
  });
});
