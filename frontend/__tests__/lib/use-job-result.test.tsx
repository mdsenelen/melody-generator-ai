import { act, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";

import { useJobResult } from "../../hooks/use-job-result";
import { RETRY_INITIAL_DELAY_MS } from "../../src/domain/retryPolicy";
import { BACKEND_API, server } from "../../src/test/msw/server";
import { renderHookWithQueryClient } from "../../src/test/query/render";

const JOB_URL = `${BACKEND_API}/transcribe/job-1`;

const TRANSCRIPTION = {
  n_notes: 3,
  duration_sec: 1,
  source_duration_sec: 1,
  truncated: false,
  midi_b64: "AAA=",
  wav_b64: null,
  midi_filename: "transcription.mid",
  wav_filename: "",
  note_events: [{ start: 0, end: 0.5, pitch: 60, velocity: 90 }],
};

function job(status: string, overrides: Record<string, unknown> = {}) {
  return {
    job_id: "job-1",
    status,
    progress: 0,
    result: null,
    error: null,
    ...overrides,
  };
}

function respondInOrder(...bodies: Array<() => Response>) {
  let calls = 0;
  server.use(
    http.get(JOB_URL, () => {
      const respond = bodies[Math.min(calls, bodies.length - 1)];
      calls += 1;
      return respond();
    }),
  );
  return () => calls;
}

const FAST_POLL = { pollIntervalMs: 10 };

describe("useJobResult", () => {
  it("polls while the job is running, then hands over the tagged result", async () => {
    const calls = respondInOrder(
      () => HttpResponse.json(job("queued")),
      () => HttpResponse.json(job("processing")),
      () => HttpResponse.json(job("completed", { result: TRANSCRIPTION })),
    );

    const { result } = renderHookWithQueryClient(() =>
      useJobResult("job-1", { pollIntervalMs: 150 }),
    );

    expect(result.current).toEqual({ status: "loading" });
    await waitFor(() => expect(result.current.status).toBe("running"));
    await waitFor(() => expect(result.current.status).toBe("succeeded"));
    expect(result.current).toEqual({
      status: "succeeded",
      result: { kind: "transcription", data: expect.objectContaining({ n_notes: 3 }) },
    });
    const settledCalls = calls();
    await new Promise((resolve) => setTimeout(resolve, 400));
    expect(calls()).toBe(settledCalls); // no polling once it's done
  });

  it("reports the job's own failure message", async () => {
    respondInOrder(() => HttpResponse.json(job("failed", { error: "Could not decode audio" })));
    const { result } = renderHookWithQueryClient(() => useJobResult("job-1", FAST_POLL));
    await waitFor(() =>
      expect(result.current).toEqual({ status: "failed", error: "Could not decode audio" }),
    );
  });

  it("fails a completed job that carries no result", async () => {
    respondInOrder(() => HttpResponse.json(job("completed")));
    const { result } = renderHookWithQueryClient(() => useJobResult("job-1", FAST_POLL));
    await waitFor(() =>
      expect(result.current).toEqual({ status: "failed", error: "Job completed with no result" }),
    );
  });

  it("reads an expired job as gone", async () => {
    respondInOrder(() => HttpResponse.json(job("expired")));
    const { result } = renderHookWithQueryClient(() => useJobResult("job-1", FAST_POLL));
    await waitFor(() => expect(result.current).toEqual({ status: "gone" }));
  });

  it("reads a 404 as gone, without retrying it", async () => {
    const calls = respondInOrder(() =>
      HttpResponse.json({ detail: "Job not found" }, { status: 404 }),
    );
    const { result } = renderHookWithQueryClient(() => useJobResult("job-1", FAST_POLL), {
      retry: true,
    });
    await waitFor(() => expect(result.current).toEqual({ status: "gone" }));
    expect(calls()).toBe(1);
  });

  it("gives up with the timeout message once the poll budget is spent", async () => {
    respondInOrder(() => HttpResponse.json(job("processing")));
    const { result } = renderHookWithQueryClient(() =>
      useJobResult("job-1", { pollIntervalMs: 10, pollMaxTotalMs: 60 }),
    );
    await waitFor(() =>
      expect(result.current).toEqual({
        status: "failed",
        error: "This is taking longer than expected. Please try again in a moment.",
      }),
    );
  });

  it("aborts the in-flight status request on unmount", async () => {
    let requestSignal: AbortSignal | undefined;
    server.use(
      http.get(JOB_URL, ({ request }) => {
        requestSignal = request.signal;
        return new Promise<Response>(() => {});
      }),
    );

    const { unmount } = renderHookWithQueryClient(() => useJobResult("job-1", FAST_POLL));
    await waitFor(() => expect(requestSignal).toBeDefined());
    unmount();

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

    it("says the server is waking while it retries a 502, then shows the result", async () => {
      respondInOrder(
        () => new HttpResponse(null, { status: 502 }),
        () => HttpResponse.json(job("completed", { result: TRANSCRIPTION })),
      );

      const { result } = renderHookWithQueryClient(() => useJobResult("job-1"), { retry: true });

      await waitFor(() => expect(result.current).toEqual({ status: "waking" }));
      await act(() => jest.advanceTimersByTimeAsync(RETRY_INITIAL_DELAY_MS));
      await waitFor(() => expect(result.current.status).toBe("succeeded"));
    });
  });
});
