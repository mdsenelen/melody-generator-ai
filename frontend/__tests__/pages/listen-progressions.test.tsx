import { render, screen, waitFor } from "@testing-library/react";

import ListenProgressionsPage from "../../app/listen-progressions/page";

function mockFetchOnce(response: { ok: boolean; status?: number; body: unknown }) {
  global.fetch = jest.fn().mockResolvedValue({
    ok: response.ok,
    status: response.status ?? (response.ok ? 200 : 500),
    statusText: "",
    headers: new Headers({ "content-type": "application/json" }),
    text: async () => JSON.stringify(response.body),
  }) as unknown as typeof fetch;
}

describe("ListenProgressionsPage", () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("renders progressions on success", async () => {
    mockFetchOnce({
      ok: true,
      body: {
        progressions: [
          {
            id: "p1",
            name: "Pop progression",
            genre: "Pop",
            source: "curated",
            chords: ["C", "G", "Am", "F"],
            song_title: null,
            artist: null,
            year: null,
          },
        ],
      },
    });

    render(<ListenProgressionsPage />);

    expect(await screen.findByText("Pop progression")).toBeInTheDocument();
  });

  it("shows a visible error instead of silently rendering an empty list when the backend fails", async () => {
    mockFetchOnce({ ok: false, status: 500, body: { detail: "backend is down" } });

    render(<ListenProgressionsPage />);

    expect(await screen.findByText(/couldn't load chord progressions/i)).toBeInTheDocument();
    expect(screen.getByText(/backend is down/i)).toBeInTheDocument();
  });

  it("shows an explicit empty state when there really are no progressions", async () => {
    mockFetchOnce({ ok: true, body: { progressions: [] } });

    render(<ListenProgressionsPage />);

    expect(await screen.findByText(/no chord progressions available yet/i)).toBeInTheDocument();
  });

  it("does not update state after unmounting mid-fetch", async () => {
    let resolveFetch: (value: unknown) => void = () => {};
    global.fetch = jest.fn(
      () =>
        new Promise((resolve) => {
          resolveFetch = resolve;
        }),
    ) as unknown as typeof fetch;

    const { unmount } = render(<ListenProgressionsPage />);
    unmount();

    // Resolve the in-flight fetch after unmount — the cancelled-flag guard
    // should prevent any setState call. If it didn't, React/RTL would flag a
    // "state update on an unmounted component" warning.
    const errorSpy = jest.spyOn(console, "error").mockImplementation(() => {});
    resolveFetch({
      ok: true,
      status: 200,
      statusText: "",
      headers: new Headers({ "content-type": "application/json" }),
      text: async () => JSON.stringify({ progressions: [] }),
    });

    await waitFor(() => {
      expect(errorSpy).not.toHaveBeenCalled();
    });
  });
});
