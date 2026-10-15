import { render, screen } from "@testing-library/react";

import Nav from "../../components/nav";

jest.mock("next/navigation", () => ({ usePathname: () => "/guide" }));

describe("Nav", () => {
  it("puts the user's guide first, left of Analyse", () => {
    render(<Nav />);
    const links = screen.getAllByRole("link");
    expect(links.map((link) => link.textContent)).toEqual([
      "User's guide",
      "Analyse",
      "Progressions",
      "Compose",
      "Variants",
    ]);
    expect(links[0]).toHaveAttribute("href", "/guide");
    expect(links[0]).toHaveAttribute("aria-current", "page");
  });
});
