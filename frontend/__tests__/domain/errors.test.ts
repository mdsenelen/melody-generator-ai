import {
  invalidInput,
  isCancellation,
  RESTARTED_MESSAGE,
  toGenerationError,
  toTranscriptionError,
} from "../../src/domain/errors";

function requestError(
  message: string,
  fields: { status?: number; code?: string; retryAfterSeconds?: number },
) {
  return Object.assign(new Error(message), fields);
}

function named(name: string, message = "x") {
  const error = new Error(message);
  error.name = name;
  return error;
}

describe("toGenerationError", () => {
  it("tells warming up apart from busy, though both are 429", () => {
    expect(
      toGenerationError(
        requestError("warming", { status: 429, code: "warming_up", retryAfterSeconds: 30 }),
      ),
    ).toEqual({ code: "warming_up", message: "warming", retryable: true, retryAfterSeconds: 30 });
    expect(toGenerationError(requestError("busy", { status: 429, retryAfterSeconds: 5 }))).toEqual({
      code: "busy",
      message: "busy",
      retryable: true,
      retryAfterSeconds: 5,
    });
  });

  it.each([[{ status: 502 }], [{ status: 503 }], [{ code: "network_error" }]])(
    "treats %j as a restart, with the reassuring copy",
    (fields) => {
      expect(toGenerationError(requestError("raw", fields))).toEqual({
        code: "restarting",
        message: RESTARTED_MESSAGE,
        retryable: true,
      });
    },
  );

  it("keeps the server's message for any other failure", () => {
    expect(toGenerationError(requestError("Model weights missing", { status: 500 }))).toEqual({
      code: "failed",
      message: "Model weights missing",
      retryable: true,
    });
  });

  it("falls back to generic copy for non-errors", () => {
    expect(toGenerationError("nope")).toEqual({
      code: "failed",
      message: "Variant generation failed",
      retryable: true,
    });
  });
});

describe("toTranscriptionError", () => {
  it("recognises a job lost to a restart", () => {
    expect(toTranscriptionError(named("TranscribeJobLostError", "lost it"))).toEqual({
      code: "job_lost",
      message: "lost it",
      retryable: true,
    });
  });

  it("keeps the raw message otherwise, even for a 503", () => {
    expect(toTranscriptionError(requestError("Bad gateway", { status: 503 }))).toEqual({
      code: "failed",
      message: "Bad gateway",
      retryable: true,
    });
    expect(toTranscriptionError(42)).toMatchObject({ message: "Transcription failed" });
  });
});

describe("invalidInput", () => {
  it("is the only non-retryable error", () => {
    expect(invalidInput("Pick a clip")).toEqual({
      code: "invalid_input",
      message: "Pick a clip",
      retryable: false,
    });
  });
});

describe("isCancellation", () => {
  it("recognises the ways a run gets cancelled", () => {
    expect(isCancellation(named("RetryCancelledError"))).toBe(true);
    expect(isCancellation(named("TranscribeJobSupersededError"))).toBe(true);
    expect(isCancellation(new DOMException("aborted", "AbortError"))).toBe(true);
  });

  it("does not swallow real failures", () => {
    expect(isCancellation(new Error("boom"))).toBe(false);
    expect(isCancellation(new DOMException("x", "NotSupportedError"))).toBe(false);
    expect(isCancellation(null)).toBe(false);
  });
});
