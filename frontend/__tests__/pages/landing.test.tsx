import { render, screen } from "@testing-library/react";

import LandingPage from "../../app/page";

describe("LandingPage", () => {
  it("keeps a page heading for screen readers without the visible headline", () => {
    render(<LandingPage />);
    expect(screen.getByRole("heading", { level: 1, name: "Melodia" })).toBeInTheDocument();
    expect(screen.queryByText(/turn a clip of audio/i)).not.toBeInTheDocument();
  });

  it("leaves the call-to-action buttons to the user's guide page", () => {
    render(<LandingPage />);
    expect(screen.queryByRole("link", { name: /start analysing audio/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /training notebook/i })).not.toBeInTheDocument();
  });

  it("leaves the how-it-works steps to the user's guide page", () => {
    render(<LandingPage />);
    expect(screen.queryByText(/how it works/i)).not.toBeInTheDocument();
  });
});
