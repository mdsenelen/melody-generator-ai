import { render, screen } from "@testing-library/react";
import { fireEvent } from "@testing-library/react";

import { Slider, TemperatureInput } from "../../../components/ui/slider";

describe("Slider", () => {
  it("exposes a labelled native range input and reports changes", () => {
    const onChange = jest.fn();
    render(<Slider label="Tempo" value={100} min={40} max={200} unit=" BPM" onChange={onChange} />);

    const slider = screen.getByRole("slider", { name: "Tempo" });
    expect(slider).toHaveAttribute("min", "40");
    expect(slider).toHaveAttribute("max", "200");
    expect(screen.getByText("100 BPM")).toBeInTheDocument();

    fireEvent.change(slider, { target: { value: "120" } });
    expect(onChange).toHaveBeenCalledWith(120);
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
