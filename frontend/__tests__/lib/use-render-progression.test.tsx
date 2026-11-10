import { act, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";

import { useRenderProgression } from "../../hooks/use-render-progression";
import { server } from "../../src/test/msw/server";
import { renderHookWithQueryClient } from "../../src/test/query/render";

const URL_PATTERN = "*/api/generate-progression";

function rendered(jobId: string) {
  return {
    progression: ["C", "G"],
    bpm: 120,
    instrument: 0,
    audio_b64: "AAA=",
    wav_b64: "AAA=",
    midi_b64: "AAA=",
    midi_filename: "progression.mid",
    midi_download_path: "/api/download/progression.mid",
    wav_filename: "progression.wav",
    wav_download_path: "/api/download/progression.wav",
    detected_chords: ["C", "G"],
    job_id: jobId,
  };
}

const REQUEST = { progression: ["C", "G"], bpm: 120, instrument: 0 };

describe("useRenderProgression", () => {
  it("renders, then keeps the parsed result", async () => {
    let body: unknown;
    server.use(
      http.post(URL_PATTERN, async ({ request }) => {
        body = await request.json();
        return HttpResponse.json(rendered("job-p1"));
      }),
    );

    const { result } = renderHookWithQueryClient(() => useRenderProgression());
    act(() => result.current.render(REQUEST));

    await waitFor(() => expect(result.current.lastResult?.job_id).toBe("job-p1"));
    expect(result.current.isRendering).toBe(false);
    expect(result.current.error).toBeNull();
    expect(body).toEqual(REQUEST);
  });

  it("surfaces a failure, keeping the previous result on screen, and never retries", async () => {
    let calls = 0;
    server.use(
      http.post(URL_PATTERN, () => {
        calls += 1;
        return calls === 1
          ? HttpResponse.json(rendered("job-p1"))
          : new HttpResponse(null, { status: 503 });
      }),
    );
    jest.spyOn(console, "error").mockImplementation(() => {});

    const { result } = renderHookWithQueryClient(() => useRenderProgression(), { retry: true });
    act(() => result.current.render(REQUEST));
    await waitFor(() => expect(result.current.lastResult?.job_id).toBe("job-p1"));

    act(() => result.current.render(REQUEST));
    await waitFor(() => expect(result.current.error).toBe("Request failed (503)"));
    expect(result.current.lastResult?.job_id).toBe("job-p1");
    expect(calls).toBe(2);
    jest.restoreAllMocks();
  });

  it("aborts the earlier request when a new render supersedes it", async () => {
    const signals: AbortSignal[] = [];
    server.use(
      http.post(URL_PATTERN, ({ request }) => {
        signals.push(request.signal);
        return signals.length === 1
          ? new Promise<Response>(() => {})
          : HttpResponse.json(rendered("job-p2"));
      }),
    );

    const { result } = renderHookWithQueryClient(() => useRenderProgression());
    act(() => result.current.render(REQUEST));
    await waitFor(() => expect(signals).toHaveLength(1));
    act(() => result.current.render({ ...REQUEST, bpm: 90 }));

    await waitFor(() => expect(result.current.lastResult?.job_id).toBe("job-p2"));
    expect(signals[0].aborted).toBe(true);
    expect(result.current.error).toBeNull();
  });

  it("aborts the request in flight on unmount", async () => {
    let requestSignal: AbortSignal | undefined;
    server.use(
      http.post(URL_PATTERN, ({ request }) => {
        requestSignal = request.signal;
        return new Promise<Response>(() => {});
      }),
    );

    const { result, unmount } = renderHookWithQueryClient(() => useRenderProgression());
    act(() => result.current.render(REQUEST));
    await waitFor(() => expect(requestSignal).toBeDefined());
    unmount();

    await waitFor(() => expect(requestSignal?.aborted).toBe(true));
  });
});
