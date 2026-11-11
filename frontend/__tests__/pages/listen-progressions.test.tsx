import { act, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";

import ListenProgressionsPage from "../../app/listen-progressions/page";
import { RETRY_INITIAL_DELAY_MS } from "../../src/domain/retryPolicy";
import { WAKING_STATUS_MESSAGE } from "../../src/domain/statusCopy";
import { server } from "../../src/test/msw/server";
import { renderWithQueryClient } from "../../src/test/query/render";

// MSW's fetch, as installed by server.listen() (jest.setup.js) -- captured
// before the tests below swap fetch for a jest.fn.
let realFetch: typeof fetch;
beforeAll(() => {
  realFetch = global.fetch;
});

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

    renderWithQueryClient(<ListenProgressionsPage />);

    expect(await screen.findByText("Pop progression")).toBeInTheDocument();
  });

  it("shows a visible error instead of silently rendering an empty list when the backend fails", async () => {
    mockFetchOnce({ ok: false, status: 500, body: { detail: "backend is down" } });

    renderWithQueryClient(<ListenProgressionsPage />);

    expect(await screen.findByText(/couldn't load chord progressions/i)).toBeInTheDocument();
    expect(screen.getByText(/backend is down/i)).toBeInTheDocument();
  });

  it("shows an explicit empty state when there really are no progressions", async () => {
    mockFetchOnce({ ok: true, body: { progressions: [] } });

    renderWithQueryClient(<ListenProgressionsPage />);

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

    const { unmount } = renderWithQueryClient(<ListenProgressionsPage />);
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

  describe("on a cold server", () => {
    beforeEach(() => {
      // The tests above swap fetch for a jest.fn; MSW needs the real one.
      global.fetch = realFetch;
      jest.useFakeTimers({ doNotFake: ["queueMicrotask", "nextTick", "setImmediate"] });
      jest.spyOn(console, "error").mockImplementation(() => {});
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it("says it's waking the server while it retries, instead of flashing an error", async () => {
      let calls = 0;
      server.use(
        http.get("*/api/progressions", () => {
          calls += 1;
          return calls === 1
            ? HttpResponse.json({ detail: "Service Unavailable" }, { status: 503 })
            : HttpResponse.json({
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
              });
        }),
      );

      renderWithQueryClient(<ListenProgressionsPage />, { retry: true });

      expect(
        await screen.findByRole("status", { name: WAKING_STATUS_MESSAGE }),
      ).toBeInTheDocument();
      expect(screen.queryByText(/couldn't load chord progressions/i)).not.toBeInTheDocument();

      await act(() => jest.advanceTimersByTimeAsync(RETRY_INITIAL_DELAY_MS));

      expect(await screen.findByText("Pop progression")).toBeInTheDocument();
    });
  });
});
