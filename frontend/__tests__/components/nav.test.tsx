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

  it.each([
    ["Analyse", "Take a look at the chords used and get a MIDI"],
    ["Progressions", "Create melodies from preset popular chord progressions"],
    ["Compose", "Compose from scratch"],
    ["Variants", "Generate alternative variants of your song"],
  ])("describes %s with a hover pop-up", (label, tip) => {
    render(<Nav />);
    const link = screen.getByRole("link", { name: label });
    const tipId = link.getAttribute("aria-describedby");
    expect(tipId).toBeTruthy();
    expect(document.getElementById(tipId ?? "")).toHaveTextContent(tip);
    expect(document.getElementById(tipId ?? "")).toHaveAttribute("role", "tooltip");
  });

  it("marks Variants as beta in its pop-up", () => {
    render(<Nav />);
    const tipId = screen.getByRole("link", { name: "Variants" }).getAttribute("aria-describedby");
    expect(document.getElementById(tipId ?? "")).toHaveTextContent(/beta/i);
  });

  it("gives the user's guide no pop-up", () => {
    render(<Nav />);
    expect(screen.getByRole("link", { name: "User's guide" })).not.toHaveAttribute(
      "aria-describedby",
    );
  });

  it("gives inactive links a hover background", () => {
    render(<Nav />);
    expect(screen.getByRole("link", { name: "Analyse" })).toHaveClass("hover:bg-raised-hover");
  });
});
