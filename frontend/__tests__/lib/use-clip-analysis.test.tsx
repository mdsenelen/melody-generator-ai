import { act, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";

import { useClipAnalysis, type ClipAnalysisRequest } from "../../hooks/use-clip-analysis";
import { RETRY_INITIAL_DELAY_MS } from "../../src/domain/retryPolicy";
import { ANALYSIS_RETRY_MESSAGE } from "../../src/domain/statusCopy";
import { BACKEND_API, server } from "../../src/test/msw/server";
import { renderHookWithQueryClient } from "../../src/test/query/render";

const ANALYZE_URL = `${BACKEND_API}/analyze`;

function analysis(overrides: Record<string, unknown> = {}) {
  return {
    tempo_bpm: 120,
    average_pitch: 61,
    mood_idx: 0,
    mood_label: "happy",
    key: "C major",
    pitch_histogram: new Array(12).fill(0.1),
    detected_chords: ["C"],
    n_notes: 3,
    clip_start_sec: 0,
    clip_end_sec: 60,
    ...overrides,
  };
}

const JOB_1: ClipAnalysisRequest = { jobId: "job-1", window: { start: 0, end: 60 } };

type Props = { request: ClipAnalysisRequest | null };

describe("useClipAnalysis", () => {
  it("stays idle with nothing to analyse, and sends nothing", () => {
    const { result } = renderHookWithQueryClient(() => useClipAnalysis(null));
    expect(result.current.state).toEqual({ status: "idle" });
  });

  it("analyses the requested window", async () => {
    const bodies: unknown[] = [];
    server.use(
      http.post(ANALYZE_URL, async ({ request }) => {
        bodies.push(await request.json());
        return HttpResponse.json(analysis());
      }),
    );

    const { result } = renderHookWithQueryClient(() => useClipAnalysis(JOB_1));

    expect(result.current.state).toEqual({ status: "loading" });
    await waitFor(() => expect(result.current.state.status).toBe("ready"));
    expect(bodies).toEqual([{ job_id: "job-1", clip_start_sec: 0, clip_end_sec: 60 }]);
  });

  it("re-analyses when the window changes", async () => {
    server.use(
      http.post(ANALYZE_URL, async ({ request }) => {
        const body = await request.json();
        const start =
          typeof body === "object" && body && "clip_start_sec" in body ? body.clip_start_sec : 0;
        return HttpResponse.json(analysis({ key: start === 10 ? "A minor" : "C major" }));
      }),
    );

    const { result, rerender } = renderHookWithQueryClient(
      ({ request }: Props) => useClipAnalysis(request),
      { initialProps: { request: JOB_1 } },
    );
    await waitFor(() => expect(result.current.state).toMatchObject({ data: { key: "C major" } }));

    rerender({ request: { jobId: "job-1", window: { start: 10, end: 20 } } });
    await waitFor(() => expect(result.current.state).toMatchObject({ data: { key: "A minor" } }));
  });

  it("shows the server's message when the analysis is rejected", async () => {
    server.use(
      http.post(ANALYZE_URL, () =>
        HttpResponse.json({ detail: "Transcription is not complete" }, { status: 409 }),
      ),
    );
    const { result } = renderHookWithQueryClient(() => useClipAnalysis(JOB_1), { retry: true });
    await waitFor(() =>
      expect(result.current.state).toEqual({
        status: "error",
        error: "Transcription is not complete",
      }),
    );
  });

  it("aborts the request when the transcription it belongs to is superseded", async () => {
    let requestSignal: AbortSignal | undefined;
    server.use(
      http.post(ANALYZE_URL, ({ request }) => {
        requestSignal = request.signal;
        return new Promise<Response>(() => {});
      }),
    );

    const { rerender } = renderHookWithQueryClient(
      ({ request }: Props) => useClipAnalysis(request),
      { initialProps: { request: JOB_1 } },
    );
    await waitFor(() => expect(requestSignal).toBeDefined());
    rerender({ request: null });

    await waitFor(() => expect(requestSignal?.aborted).toBe(true));
  });

  describe("on a cold server", () => {
    beforeEach(() => {
      jest.useFakeTimers({ doNotFake: ["queueMicrotask", "nextTick", "setImmediate"] });
      jest.spyOn(console, "error").mockImplementation(() => {});
    });

    afterEach(() => {
      jest.useRealTimers();
      jest.restoreAllMocks();
    });

    it("keeps loading with the waking message while it retries", async () => {
      let calls = 0;
      server.use(
        http.post(ANALYZE_URL, () => {
          calls += 1;
          return calls === 1
            ? new HttpResponse(null, { status: 503 })
            : HttpResponse.json(analysis());
        }),
      );

      const { result } = renderHookWithQueryClient(() => useClipAnalysis(JOB_1), {
        retry: true,
      });

      await waitFor(() =>
        expect(result.current.state).toEqual({
          status: "loading",
          message: ANALYSIS_RETRY_MESSAGE,
        }),
      );
      await act(() => jest.advanceTimersByTimeAsync(RETRY_INITIAL_DELAY_MS));
      await waitFor(() => expect(result.current.state.status).toBe("ready"));
    });
  });
});
