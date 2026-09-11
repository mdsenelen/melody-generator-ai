import { render, screen } from "@testing-library/react";

import { Button } from "../../../components/ui/button";

describe("Button", () => {
  it("renders a real, enabled button by default", () => {
    render(<Button>Generate</Button>);
    const button = screen.getByRole("button", { name: "Generate" });
    expect(button.tagName).toBe("BUTTON");
    expect(button).toBeEnabled();
  });

  it("disables the button while loading, without hiding its label", () => {
    render(<Button loading>Generate</Button>);
    expect(screen.getByRole("button", { name: "Generate" })).toBeDisabled();
  });

  it("respects an explicit disabled prop", () => {
    render(<Button disabled>Generate</Button>);
    expect(screen.getByRole("button", { name: "Generate" })).toBeDisabled();
  });
});
