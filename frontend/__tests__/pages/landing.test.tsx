import { render, screen } from "@testing-library/react";

import LandingPage from "../../app/page";

describe("LandingPage", () => {
  it("renders the hero heading", () => {
    render(<LandingPage />);
    expect(
      screen.getByRole("heading", {
        name: /turn your humming into a new melody, or analyse an audio clip/i,
      }),
    ).toBeInTheDocument();
  });

  it("does not render the removed AI music generation tag", () => {
    render(<LandingPage />);
    expect(screen.queryByText(/ai music generation/i)).not.toBeInTheDocument();
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

  it("drops the Good to know notes and the Coming soon list", () => {
    render(<LandingPage />);
    expect(screen.queryByText(/good to know/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/cold-server latency/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/coming soon/i)).not.toBeInTheDocument();
    expect(screen.queryByText("Sharing & collaboration")).not.toBeInTheDocument();
  });
});
