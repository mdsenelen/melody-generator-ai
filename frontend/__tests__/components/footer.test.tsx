import { render, screen } from "@testing-library/react";

import Footer from "../../components/footer";

describe("Footer", () => {
  it("is set in the title face, like the logo and headings", () => {
    render(<Footer />);
    expect(screen.getByRole("contentinfo")).toHaveClass("font-title");
  });
});
