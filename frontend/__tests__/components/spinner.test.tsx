import { render, screen } from "@testing-library/react";

import { Spinner } from "../../components/spinner";

describe("Spinner", () => {
  it("has an accessible name from its label and disables the spin animation under prefers-reduced-motion", () => {
    render(<Spinner label="Uploading" />);

    const status = screen.getByRole("status", { name: "Uploading" });
    const glyph = status.querySelector("[aria-hidden='true']");

    expect(glyph).toHaveClass("animate-spin");
    expect(glyph).toHaveClass("motion-reduce:animate-none");
  });
});
