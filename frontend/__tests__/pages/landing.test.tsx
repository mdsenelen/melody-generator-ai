import { render, screen } from "@testing-library/react";

import LandingPage from "../../app/page";

describe("LandingPage", () => {
  it("renders the hero heading", () => {
    render(<LandingPage />);
    expect(
      screen.getByRole("heading", { name: /turn a clip of audio into new, playable melodies/i }),
    ).toBeInTheDocument();
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
