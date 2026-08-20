import {
  createTranscribeJob,
  getTranscribeJob,
  pollTranscribeJob,
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

const COMPLETED_RESULT = {
  n_notes: 4,
  duration_sec: 1.5,
  source_duration_sec: 1.5,
  truncated: false,
  midi_b64: "AAA=",
  wav_b64: null,
  midi_filename: "t.mid",
  wav_filename: "",
  mood_label: "happy" as const,
  mood_idx: 0,
  detected_chords: ["C"],
  key: "C major",
  pitch_histogram: [0.1],
  tempo_bpm: 120,
  average_pitch: 61,
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
    expect(result.key).toBe("C major");
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
    expect(result.key).toBe("C major");
  });

  it("gives up after too many consecutive poll failures", async () => {
    mockFetch().mockRejectedValue(new Error("network down"));

    const promise = pollTranscribeJob("job-1", { isSuperseded: () => false });
    const expectation = expect(promise).rejects.toThrow("network down");
    for (let i = 0; i < 5; i += 1) {
      await Promise.resolve();
      await jest.advanceTimersByTimeAsync(5000);
    }
    await expectation;
  });

  it("throws TranscribeJobTimeoutError once the total poll budget is exceeded", async () => {
    mockFetch().mockResolvedValue(jsonResponse(200, statusResponse({ status: "processing" })));

    const promise = pollTranscribeJob("job-1", { isSuperseded: () => false });
    const expectation = expect(promise).rejects.toBeInstanceOf(TranscribeJobTimeoutError);

    // POLL_MAX_TOTAL_MS is 15 minutes (matches the backend worker's lease
    // margin above a long clip's worst-case transcription time -- see
    // transcribeJob.ts) -- 100 * 10000ms = 1,000,000ms comfortably exceeds
    // the 900,000ms budget.
    for (let i = 0; i < 100; i += 1) {
      await Promise.resolve();
      await jest.advanceTimersByTimeAsync(10000);
    }
    await expectation;
  });
});
