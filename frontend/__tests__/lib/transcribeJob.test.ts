import {
  createTranscribeJob,
  getTranscribeJob,
  POLL_MAX_TOTAL_MS,
  pollTranscribeJob,
  TranscribeJobLostError,
  TranscribeJobSupersededError,
  TranscribeJobTimeoutError,
  type TranscribeJobStatusResponse,
} from "../../app/lib/transcribeJob";

function jsonResponse(status: number, body: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (name: string) => (name === "content-type" ? "application/json" : "") },
    text: async () => JSON.stringify(body),
  } as Response;
}

// A bare HTTP error with no JSON body -- matches Render's own raw platform
// 502 (nothing listening yet) more closely than a crafted JSON error would.
function bareErrorResponse(status: number) {
  return {
    ok: false,
    status,
    headers: { get: () => "" },
    text: async () => "",
  } as Response;
}

const COMPLETED_RESULT = {
  n_notes: 4,
  duration_sec: 1.5,
  source_duration_sec: 1.5,
  truncated: false,
  midi_b64: "AAA=",
  wav_b64: null,
  midi_filename: "t.mid",
  wav_filename: "",
  n_chunks: 1,
  note_events: [{ start: 0, end: 0.5, pitch: 60, velocity: 90 }],
};

function statusResponse(
  overrides: Partial<TranscribeJobStatusResponse>,
): TranscribeJobStatusResponse {
  return {
    job_id: "job-1",
    status: "queued",
    progress: 0,
    result: null,
    error: null,
    ...overrides,
  };
}

// jsdom doesn't polyfill fetch, so `global.fetch` doesn't exist for
// jest.spyOn to wrap -- define it as a mock directly instead.
beforeEach(() => {
  global.fetch = jest.fn();
});

function mockFetch() {
  return global.fetch as jest.Mock;
}

describe("createTranscribeJob", () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("posts the file as multipart form data and returns the parsed job", async () => {
    const fetchMock = mockFetch().mockResolvedValue(
      jsonResponse(202, statusResponse({ status: "queued" })),
    );

    const file = new File(["audio-bytes"], "clip.wav", { type: "audio/wav" });
    const result = await createTranscribeJob(file);

    expect(result.status).toBe("queued");
    expect(result.job_id).toBe("job-1");
    const [, options] = fetchMock.mock.calls[0];
    expect(options?.method).toBe("POST");
    expect(options?.body).toBeInstanceOf(FormData);
  });

  it("references an existing upload by id instead of re-sending the file when one is provided", async () => {
    const fetchMock = mockFetch().mockResolvedValue(
      jsonResponse(202, statusResponse({ status: "queued" })),
    );

    const file = new File(["audio-bytes"], "clip.wav", { type: "audio/wav" });
    await createTranscribeJob(file, { id: "up-1", filename: "upload_up-1.wav" });

    const [, options] = fetchMock.mock.calls[0];
    const body = options?.body as FormData;
    expect(body.get("upload_id")).toBe("up-1");
    expect(body.get("filename")).toBe("upload_up-1.wav");
    // The whole point: no second copy of the file bytes goes over the wire.
    expect(body.get("file")).toBeNull();
  });
});

describe("getTranscribeJob", () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("GETs the job status endpoint by id", async () => {
    const fetchMock = mockFetch().mockResolvedValue(
      jsonResponse(200, statusResponse({ status: "processing", progress: 50 })),
    );

    const result = await getTranscribeJob("job-1");

    expect(result.status).toBe("processing");
    const [url] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/transcribe/job-1");
  });
});

describe("pollTranscribeJob", () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.restoreAllMocks();
  });

  it("polls until completion and returns the result", async () => {
    mockFetch()
      .mockResolvedValueOnce(jsonResponse(200, statusResponse({ status: "queued" })))
      .mockResolvedValueOnce(
        jsonResponse(200, statusResponse({ status: "processing", progress: 50 })),
      )
      .mockResolvedValueOnce(
        jsonResponse(
          200,
          statusResponse({ status: "completed", progress: 100, result: COMPLETED_RESULT }),
        ),
      );

    const onStatusChange = jest.fn();
    const promise = pollTranscribeJob("job-1", { isSuperseded: () => false, onStatusChange });

    // Flush each poll iteration's fetch + backoff sleep.
    for (let i = 0; i < 3; i += 1) {
      await Promise.resolve();
      await jest.advanceTimersByTimeAsync(5000);
    }

    const result = await promise;
    expect(result.n_notes).toBe(4);
    expect(onStatusChange).toHaveBeenCalledTimes(3);
  });

  it("rejects with the job's error message when the job fails", async () => {
    mockFetch().mockResolvedValueOnce(
      jsonResponse(200, statusResponse({ status: "failed", error: "Could not decode audio" })),
    );

    const promise = pollTranscribeJob("job-1", { isSuperseded: () => false });
    await Promise.resolve();

    await expect(promise).rejects.toThrow("Could not decode audio");
  });

  it("stops polling once superseded and throws TranscribeJobSupersededError", async () => {
    const fetchMock = mockFetch().mockResolvedValue(
      jsonResponse(200, statusResponse({ status: "queued" })),
    );

    let superseded = false;
    const promise = pollTranscribeJob("job-1", { isSuperseded: () => superseded });

    await Promise.resolve();
    superseded = true;
    await jest.advanceTimersByTimeAsync(5000);

    await expect(promise).rejects.toBeInstanceOf(TranscribeJobSupersededError);
    const callsAfterSupersession = fetchMock.mock.calls.length;
    await jest.advanceTimersByTimeAsync(20000);
    expect(fetchMock.mock.calls.length).toBe(callsAfterSupersession);
  });

  it("tolerates a couple of transient poll failures before recovering", async () => {
    mockFetch()
      .mockRejectedValueOnce(new Error("network blip"))
      .mockRejectedValueOnce(new Error("network blip"))
      .mockResolvedValueOnce(
        jsonResponse(200, statusResponse({ status: "completed", result: COMPLETED_RESULT })),
      );

    const promise = pollTranscribeJob("job-1", { isSuperseded: () => false });
    for (let i = 0; i < 3; i += 1) {
      await Promise.resolve();
      await jest.advanceTimersByTimeAsync(5000);
    }

    const result = await promise;
    expect(result.n_notes).toBe(4);
  });

  it("gives up after too many consecutive poll failures that aren't the restart signature", async () => {
    // 500 isn't 502/503/a network error -- it's our own catch-all handler
    // reporting a genuine unexpected failure, not a Render restart, so this
    // must still hit the fast consecutive-failure cutoff rather than retry
    // for the full POLL_MAX_TOTAL_MS budget.
    mockFetch().mockResolvedValue(bareErrorResponse(500));

    const promise = pollTranscribeJob("job-1", { isSuperseded: () => false });
    const expectation = expect(promise).rejects.toThrow("Request failed (500)");
    for (let i = 0; i < 5; i += 1) {
      await Promise.resolve();
      await jest.advanceTimersByTimeAsync(5000);
    }
    await expectation;
  });

  it("treats 502s, 503s, and network errors as a restart and keeps polling without exhausting the retry budget", async () => {
    mockFetch()
      .mockResolvedValueOnce(bareErrorResponse(502))
      .mockResolvedValueOnce(bareErrorResponse(503))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(bareErrorResponse(502))
      .mockResolvedValueOnce(
        jsonResponse(200, statusResponse({ status: "completed", result: COMPLETED_RESULT })),
      );

    const onRestarting = jest.fn();
    const promise = pollTranscribeJob("job-1", { isSuperseded: () => false, onRestarting });

    for (let i = 0; i < 5; i += 1) {
      await Promise.resolve();
      await jest.advanceTimersByTimeAsync(5000);
    }

    const result = await promise;
    expect(result.n_notes).toBe(4);
    // All four transient failures hit onRestarting -- none of them counted
    // toward POLL_MAX_CONSECUTIVE_FAILURES (3), which this sequence would
    // otherwise have tripped.
    expect(onRestarting).toHaveBeenCalledTimes(4);
  });

  it("fails fast with TranscribeJobLostError on a 404, without retrying", async () => {
    const fetchMock = mockFetch().mockResolvedValueOnce(bareErrorResponse(404));

    const promise = pollTranscribeJob("job-1", { isSuperseded: () => false });
    await Promise.resolve();

    await expect(promise).rejects.toBeInstanceOf(TranscribeJobLostError);
    expect(fetchMock.mock.calls.length).toBe(1);
  });

  it("keeps polling through a warming_up wait without timing out, until the job completes", async () => {
    // The backend can bounce a job queued for up to WARMUP_MAX_SECONDS
    // (240s, backend/app/inference.py) without ever marking it failed --
    // the client-side poll must survive that whole window, well short of
    // POLL_MAX_TOTAL_MS (900s, see the timeout test below).
    mockFetch()
      .mockResolvedValueOnce(
        jsonResponse(200, statusResponse({ status: "queued", waiting_reason: "warming_up" })),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, statusResponse({ status: "queued", waiting_reason: "warming_up" })),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, statusResponse({ status: "completed", result: COMPLETED_RESULT })),
      );

    const onStatusChange = jest.fn();
    const promise = pollTranscribeJob("job-1", { isSuperseded: () => false, onStatusChange });

    for (let i = 0; i < 3; i += 1) {
      await Promise.resolve();
      await jest.advanceTimersByTimeAsync(5000);
    }

    const result = await promise;
    expect(result.n_notes).toBe(4);
    expect(onStatusChange).toHaveBeenCalledWith(
      expect.objectContaining({ status: "queued", waiting_reason: "warming_up" }),
      expect.any(Number),
    );
  });

  it("throws TranscribeJobTimeoutError once the total poll budget is exceeded", async () => {
    mockFetch().mockResolvedValue(jsonResponse(200, statusResponse({ status: "processing" })));

    const promise = pollTranscribeJob("job-1", { isSuperseded: () => false });
    const expectation = expect(promise).rejects.toBeInstanceOf(TranscribeJobTimeoutError);

    // 130 * 10000ms = 1,300,000ms comfortably exceeds POLL_MAX_TOTAL_MS
    // (see its own definition in transcribeJob.ts for the arithmetic this
    // budget is sized against) regardless of exactly how large that
    // budget is, so this test doesn't need updating every time the
    // budget itself is retuned.
    const iterations = Math.ceil((POLL_MAX_TOTAL_MS + 100000) / 10000);
    for (let i = 0; i < iterations; i += 1) {
      await Promise.resolve();
      await jest.advanceTimersByTimeAsync(10000);
    }
    await expectation;
  });

  it("covers the worst-case recovery chain: dead-worker detection + a full warm-up + the longest allowed transcription, with margin", () => {
    // Pins the actual budget against the real arithmetic, rather than
    // only a generic "it eventually times out" check above. Mirrors
    // CLAUDE.md's "poisoned floor" recovery-chain numbers and
    // jobs/worker.py's DEFAULT_LEASE_SECONDS -- if any of those move,
    // this is the number that needs re-deriving, not silently left stale.
    const leaseDetectionWorstCaseSec = 120 + 5; // DEFAULT_LEASE_SECONDS + one idle-tick interval
    const warmupBudgetSec = 240; // WARMUP_MAX_SECONDS
    const longestTranscriptionSec = 600; // MAX_UPLOAD_DURATION_SEC, ~real-time
    const minimumRequiredSec =
      leaseDetectionWorstCaseSec + warmupBudgetSec + longestTranscriptionSec;

    expect(POLL_MAX_TOTAL_MS).toBeGreaterThanOrEqual(minimumRequiredSec * 1000);
    // Real margin on top, not just barely clearing the minimum.
    expect(POLL_MAX_TOTAL_MS - minimumRequiredSec * 1000).toBeGreaterThanOrEqual(120 * 1000);
  });
});
