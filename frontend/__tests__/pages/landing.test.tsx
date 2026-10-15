import { render, screen } from "@testing-library/react";

import LandingPage from "../../app/page";

describe("LandingPage", () => {
  it("keeps a page heading for screen readers without the visible headline", () => {
    render(<LandingPage />);
    expect(screen.getByRole("heading", { level: 1, name: "Melodia" })).toBeInTheDocument();
    expect(screen.queryByText(/turn a clip of audio/i)).not.toBeInTheDocument();
  });

  it("links its primary call to action to /analyse", () => {
    render(<LandingPage />);
    expect(screen.getByRole("link", { name: /start analysing audio/i })).toHaveAttribute(
      "href",
      "/analyse",
    );
  });

  it("links every pipeline step to an existing route", () => {
    render(<LandingPage />);
    const expectedRoutes = ["/analyse", "/choose-progression", "/generate-variants"];
    for (const route of expectedRoutes) {
      expect(screen.getAllByRole("link").some((link) => link.getAttribute("href") === route)).toBe(
        true,
      );
    }
  });
});
