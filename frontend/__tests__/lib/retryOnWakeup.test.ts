import {
  RETRY_ATTEMPT_TIMEOUT_MS,
  RETRY_INITIAL_DELAY_MS,
  RETRY_MAX_TOTAL_MS,
  RetryCancelledError,
  retryOnWakeup,
} from "../../app/lib/retryOnWakeup";

function networkError() {
  const error = new Error("net") as Error & { code?: string };
  error.code = "network_error";
  return error;
}

function httpError(status: number) {
  const error = new Error(`Request failed (${status})`) as Error & { status?: number };
  error.status = status;
  return error;
}

function hangingAttempt() {
  return (signal: AbortSignal) =>
    new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    });
}

describe("retryOnWakeup", () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("returns the result on the first successful attempt without retrying", async () => {
    const fn = jest.fn().mockResolvedValue("ok");

    const result = await retryOnWakeup(fn);

    expect(result).toBe("ok");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("retries a network_error with backoff and reports the retry before succeeding", async () => {
    const fn = jest.fn().mockRejectedValueOnce(networkError()).mockResolvedValueOnce("ok");
    const onRetrying = jest.fn();

    const promise = retryOnWakeup(fn, { onRetrying });
    await Promise.resolve();
    await jest.advanceTimersByTimeAsync(RETRY_INITIAL_DELAY_MS);

    await expect(promise).resolves.toBe("ok");
    expect(fn).toHaveBeenCalledTimes(2);
    expect(onRetrying).toHaveBeenCalledTimes(1);
  });

  it.each([502, 503])("retries a bare %i the same way as a network error", async (status) => {
    const fn = jest.fn().mockRejectedValueOnce(httpError(status)).mockResolvedValueOnce("ok");

    const promise = retryOnWakeup(fn);
    await Promise.resolve();
    await jest.advanceTimersByTimeAsync(RETRY_INITIAL_DELAY_MS);

    await expect(promise).resolves.toBe("ok");
  });

  it("does not retry an error that isn't the wake-up signature", async () => {
    const fn = jest.fn().mockRejectedValue(httpError(500));

    await expect(retryOnWakeup(fn)).rejects.toMatchObject({ status: 500 });
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("treats an attempt that never settles as a hang and retries once the per-attempt timeout fires", async () => {
    const fn = jest.fn().mockImplementationOnce(hangingAttempt()).mockResolvedValueOnce("ok");

    const promise = retryOnWakeup(fn);
    await Promise.resolve();
    await jest.advanceTimersByTimeAsync(RETRY_ATTEMPT_TIMEOUT_MS);
    await jest.advanceTimersByTimeAsync(RETRY_INITIAL_DELAY_MS);

    await expect(promise).resolves.toBe("ok");
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("stops retrying once the total budget is exhausted and rethrows the last error", async () => {
    const fn = jest.fn().mockRejectedValue(networkError());

    const promise = retryOnWakeup(fn);
    const expectation = expect(promise).rejects.toMatchObject({ code: "network_error" });
    await jest.advanceTimersByTimeAsync(RETRY_MAX_TOTAL_MS + RETRY_ATTEMPT_TIMEOUT_MS);

    await expectation;
  });

  it("stops immediately once the signal is aborted, without waiting out the retry delay", async () => {
    const controller = new AbortController();
    const fn = jest.fn().mockRejectedValue(networkError());

    const promise = retryOnWakeup(fn, { signal: controller.signal });
    // Let the first attempt's rejection propagate into the backoff sleep
    // before aborting, so this actually exercises a mid-delay cancellation
    // rather than a pre-aborted signal.
    await jest.advanceTimersByTimeAsync(0);
    controller.abort();

    await expect(promise).rejects.toBeInstanceOf(RetryCancelledError);
    expect(fn).toHaveBeenCalledTimes(1);
  });
});
