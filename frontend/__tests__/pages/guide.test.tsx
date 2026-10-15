import { render, screen } from "@testing-library/react";

import GuidePage from "../../app/guide/page";

describe("GuidePage (User's guide)", () => {
  it("explains how it works", () => {
    render(<GuidePage />);
    expect(screen.getByRole("heading", { level: 1, name: "How it works" })).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(4);
  });

  it("links every step to an existing route", () => {
    render(<GuidePage />);
    const hrefs = screen.getAllByRole("link").map((link) => link.getAttribute("href"));
    for (const route of ["/analyse", "/choose-progression", "/generate-variants"]) {
      expect(hrefs).toContain(route);
    }
  });

  it("ends with the calls to action that used to be on the home page", () => {
    render(<GuidePage />);
    expect(screen.getByRole("link", { name: /start analysing audio/i })).toHaveAttribute(
      "href",
      "/analyse",
    );
    expect(screen.getByRole("link", { name: /view the training notebook/i })).toHaveAttribute(
      "href",
      expect.stringContaining("colab.research.google.com"),
    );
  });
});
