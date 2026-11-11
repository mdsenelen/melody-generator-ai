import { act, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";

import { useProgressions } from "../../hooks/use-progressions";
import { RETRY_INITIAL_DELAY_MS } from "../../src/domain/retryPolicy";
import { server } from "../../src/test/msw/server";
import { renderHookWithQueryClient } from "../../src/test/query/render";

const PROGRESSION = {
  id: "pop-1",
  name: "I–V–vi–IV",
  genre: "Pop",
  source: "musicca",
  chords: ["C", "G", "Am", "F"],
  song_title: null,
  artist: null,
  year: null,
};

describe("useProgressions", () => {
  it("loads, then hands over the parsed list", async () => {
    server.use(
      http.get("*/api/progressions", () => HttpResponse.json({ progressions: [PROGRESSION] })),
    );

    const { result } = renderHookWithQueryClient(() => useProgressions());

    expect(result.current).toEqual({ status: "loading" });
    await waitFor(() =>
      expect(result.current).toEqual({ status: "ready", progressions: [PROGRESSION] }),
    );
  });

  it("fails with the server's message on a non-transient error", async () => {
    server.use(
      http.get("*/api/progressions", () =>
        HttpResponse.json({ detail: "backend is down" }, { status: 500 }),
      ),
    );

    const { result } = renderHookWithQueryClient(() => useProgressions(), { retry: true });

    await waitFor(() =>
      expect(result.current).toEqual({ status: "failed", error: "backend is down" }),
    );
  });

  it("fails on a body that breaks the contract, without retrying it", async () => {
    let calls = 0;
    server.use(
      http.get("*/api/progressions", () => {
        calls += 1;
        return HttpResponse.json({ progressions: [{ id: "x" }] });
      }),
    );
    jest.spyOn(console, "error").mockImplementation(() => {});

    const { result } = renderHookWithQueryClient(() => useProgressions(), { retry: true });

    await waitFor(() => expect(result.current.status).toBe("failed"));
    expect(calls).toBe(1);
    jest.restoreAllMocks();
  });

  it("aborts the request when the component unmounts", async () => {
    let requestSignal: AbortSignal | undefined;
    server.use(
      http.get("*/api/progressions", ({ request }) => {
        requestSignal = request.signal;
        return new Promise<Response>(() => {});
      }),
    );

    const { unmount } = renderHookWithQueryClient(() => useProgressions());
    await waitFor(() => expect(requestSignal).toBeDefined());
    unmount();

    await waitFor(() => expect(requestSignal?.aborted).toBe(true));
  });

  describe("on a cold server", () => {
    beforeEach(() => {
      jest.useFakeTimers({ doNotFake: ["queueMicrotask", "nextTick", "setImmediate"] });
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it("says it's waking the server while it retries, instead of failing", async () => {
      let calls = 0;
      server.use(
        http.get("*/api/progressions", () => {
          calls += 1;
          return calls === 1
            ? new HttpResponse(null, { status: 503 })
            : HttpResponse.json({ progressions: [PROGRESSION] });
        }),
      );
      jest.spyOn(console, "error").mockImplementation(() => {});

      const { result } = renderHookWithQueryClient(() => useProgressions(), { retry: true });

      await waitFor(() => expect(result.current).toEqual({ status: "waking" }));
      await act(() => jest.advanceTimersByTimeAsync(RETRY_INITIAL_DELAY_MS));
      await waitFor(() => expect(result.current.status).toBe("ready"));
      expect(calls).toBe(2);
      jest.restoreAllMocks();
    });
  });
});
