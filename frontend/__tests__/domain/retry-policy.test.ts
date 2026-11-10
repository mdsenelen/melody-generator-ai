import {
  backoffDelayMs,
  classifyFailure,
  isRetryingTransient,
  isTransientFailure,
  retryDelayMs,
  RETRY_INITIAL_DELAY_MS,
  RETRY_MAX_DELAY_MS,
  RETRY_MAX_TOTAL_MS,
  shouldRetry,
} from "../../src/domain/retryPolicy";

function httpError(status: number, extra: Record<string, unknown> = {}) {
  return Object.assign(new Error(`Request failed (${status})`), { status, ...extra });
}

function coded(code: string) {
  return Object.assign(new Error(code), { code });
}

function named(name: string) {
  const error = new Error(name);
  error.name = name;
  return error;
}

describe("classifyFailure", () => {
  it.each([
    ["a network error", coded("network_error"), "transient"],
    ["a hung attempt", coded("attempt_timeout"), "transient"],
    ["a 502", httpError(502), "transient"],
    ["a 503", httpError(503), "transient"],
    ["a 400", httpError(400), "rejected"],
    ["a 404", httpError(404), "rejected"],
    ["a 409", httpError(409), "rejected"],
    ["a 413", httpError(413), "rejected"],
    ["a 415", httpError(415), "rejected"],
    ["a 422", httpError(422), "rejected"],
    ["a 429 busy", httpError(429), "busy"],
    ["a 429 warming_up", httpError(429, { code: "warming_up" }), "busy"],
    ["a contract violation", named("ContractError"), "contract"],
    ["a cancelled retry", named("RetryCancelledError"), "cancelled"],
    ["a superseded poll", named("TranscribeJobSupersededError"), "cancelled"],
    ["an aborted fetch", new DOMException("The operation was aborted.", "AbortError"), "cancelled"],
    ["a 500", httpError(500), "permanent"],
    ["a plain Error", new Error("boom"), "permanent"],
    ["a non-Error value", "boom", "permanent"],
  ])("%s is %s", (_label, error, expected) => {
    expect(classifyFailure(error)).toBe(expected);
  });

  it("treats a contract violation as non-retryable even with a transient-looking code", () => {
    const error = Object.assign(named("ContractError"), { status: 503 });
    expect(classifyFailure(error)).toBe("contract");
  });
});

describe("isTransientFailure", () => {
  it("is the single definition of 'the server is waking up'", () => {
    expect(isTransientFailure(coded("network_error"))).toBe(true);
    expect(isTransientFailure(httpError(503))).toBe(true);
    expect(isTransientFailure(httpError(429))).toBe(false);
    expect(isTransientFailure(undefined)).toBe(false);
  });
});

describe("backoffDelayMs", () => {
  it("doubles from the initial delay and caps at the max", () => {
    expect([0, 1, 2, 3, 4, 5].map(backoffDelayMs)).toEqual([
      RETRY_INITIAL_DELAY_MS,
      RETRY_INITIAL_DELAY_MS * 2,
      RETRY_INITIAL_DELAY_MS * 4,
      RETRY_MAX_DELAY_MS,
      RETRY_MAX_DELAY_MS,
      RETRY_MAX_DELAY_MS,
    ]);
  });
});

describe("retryDelayMs", () => {
  it("uses the backoff when the server gave no retry-after", () => {
    expect(retryDelayMs(0, httpError(503))).toBe(RETRY_INITIAL_DELAY_MS);
    expect(retryDelayMs(2, httpError(503))).toBe(RETRY_INITIAL_DELAY_MS * 4);
  });

  it("honours retry-after when the server sent one", () => {
    expect(retryDelayMs(0, httpError(503, { retryAfterSeconds: 7 }))).toBe(7_000);
  });

  it("never waits longer than the whole budget for one retry-after", () => {
    expect(retryDelayMs(0, httpError(503, { retryAfterSeconds: 3600 }))).toBe(RETRY_MAX_TOTAL_MS);
  });
});

describe("shouldRetry", () => {
  it.each([
    ["network error", coded("network_error"), true],
    ["hung attempt", coded("attempt_timeout"), true],
    ["502", httpError(502), true],
    ["503", httpError(503), true],
    ["400", httpError(400), false],
    ["404", httpError(404), false],
    ["413", httpError(413), false],
    ["415", httpError(415), false],
    ["422", httpError(422), false],
    ["429 busy", httpError(429), false],
    ["429 warming_up", httpError(429, { code: "warming_up" }), false],
    ["500", httpError(500), false],
    ["contract violation", named("ContractError"), false],
    ["cancellation", named("RetryCancelledError"), false],
  ])("first failure: %s -> %s", (_label, error, expected) => {
    expect(shouldRetry(0, error)).toBe(expected);
  });

  it("keeps retrying a transient failure until the backoff would overrun the budget", () => {
    const error = coded("network_error");
    let failureCount = 0;
    let waited = 0;
    while (shouldRetry(failureCount, error)) {
      waited += retryDelayMs(failureCount, error);
      failureCount += 1;
    }
    expect(failureCount).toBeGreaterThan(3);
    expect(waited).toBeLessThan(RETRY_MAX_TOTAL_MS);
    expect(waited + backoffDelayMs(failureCount)).toBeGreaterThanOrEqual(RETRY_MAX_TOTAL_MS);
  });
});

describe("isRetryingTransient", () => {
  it("is true only while a fetch is in flight after a transient failure", () => {
    const transient = coded("network_error");
    expect(
      isRetryingTransient({ fetchStatus: "fetching", failureCount: 1, failureReason: transient }),
    ).toBe(true);
    expect(
      isRetryingTransient({ fetchStatus: "idle", failureCount: 1, failureReason: transient }),
    ).toBe(false);
    expect(
      isRetryingTransient({ fetchStatus: "fetching", failureCount: 0, failureReason: null }),
    ).toBe(false);
    expect(
      isRetryingTransient({
        fetchStatus: "fetching",
        failureCount: 1,
        failureReason: httpError(500),
      }),
    ).toBe(false);
  });
});
