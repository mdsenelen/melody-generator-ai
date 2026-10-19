import { RetryCancelledError, RETRY_INITIAL_DELAY_MS } from "../../app/lib/retryOnWakeup";
import {
  WAIT_ATTEMPT_TIMEOUT_MS,
  WAIT_MAX_TOTAL_MS,
  waitForBackend,
} from "../../app/lib/waitForBackend";

function okResponse(status = 200) {
  return { ok: status < 400, status } as Response;
}

function hangingFetch() {
  return (_input: RequestInfo | URL, init?: RequestInit) =>
    new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () =>
        reject(new DOMException("aborted", "AbortError")),
      );
    });
}

describe("waitForBackend", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
    global.fetch = originalFetch;
  });

  it("pings the bare /health route, not one under /api", async () => {
    const fetchMock = jest.fn().mockResolvedValue(okResponse());
    global.fetch = fetchMock;

    await waitForBackend();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toMatch(/\/health$/);
    expect(url).not.toMatch(/\/api\/health$/);
  });

  it("resolves once the server answers after N failed pings, reporting that it's waiting", async () => {
    const fetchMock = jest
      .fn()
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(okResponse(503))
      .mockResolvedValueOnce(okResponse());
    global.fetch = fetchMock;
    const onWaiting = jest.fn();

    const promise = waitForBackend({ onWaiting });
    await jest.advanceTimersByTimeAsync(RETRY_INITIAL_DELAY_MS * 3);

    await expect(promise).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(onWaiting).toHaveBeenCalled();
  });

  it("treats any non-502/503 answer as awake -- the server responded", async () => {
    global.fetch = jest.fn().mockResolvedValue(okResponse(500));

    await expect(waitForBackend()).resolves.toBeUndefined();
  });

  it("covers a multi-minute wake, well past the old 90s retry budget", async () => {
    let calls = 0;
    global.fetch = jest.fn().mockImplementation(() => {
      calls += 1;
      return Date.now() < 180_000 + startedAt
        ? Promise.reject(new TypeError("Failed to fetch"))
        : Promise.resolve(okResponse());
    });
    const startedAt = Date.now();

    const promise = waitForBackend();
    await jest.advanceTimersByTimeAsync(200_000);

    await expect(promise).resolves.toBeUndefined();
    expect(calls).toBeGreaterThan(1);
  });

  it("rethrows the last error once the total budget is exhausted", async () => {
    global.fetch = jest.fn().mockResolvedValue(okResponse(503));

    const promise = waitForBackend();
    const assertion = expect(promise).rejects.toMatchObject({ status: 503 });
    await jest.advanceTimersByTimeAsync(WAIT_MAX_TOTAL_MS + WAIT_ATTEMPT_TIMEOUT_MS);

    await assertion;
  });

  it("reports a hung ping as a network_error, not a raw AbortError, when the budget runs out", async () => {
    global.fetch = jest.fn().mockImplementation(hangingFetch());

    const promise = waitForBackend();
    const assertion = expect(promise).rejects.toMatchObject({ code: "network_error" });
    await jest.advanceTimersByTimeAsync(WAIT_MAX_TOTAL_MS + WAIT_ATTEMPT_TIMEOUT_MS * 2);

    await assertion;
  });

  it("rejects immediately when the caller aborts mid-backoff", async () => {
    const fetchMock = jest.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    global.fetch = fetchMock;
    const controller = new AbortController();

    const promise = waitForBackend({ signal: controller.signal });
    const assertion = expect(promise).rejects.toBeInstanceOf(RetryCancelledError);
    await jest.advanceTimersByTimeAsync(RETRY_INITIAL_DELAY_MS / 2);
    controller.abort();

    await assertion;
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("rejects immediately when the caller aborts mid-ping", async () => {
    global.fetch = jest.fn().mockImplementation(hangingFetch());
    const controller = new AbortController();

    const promise = waitForBackend({ signal: controller.signal });
    const assertion = expect(promise).rejects.toBeInstanceOf(RetryCancelledError);
    await jest.advanceTimersByTimeAsync(1_000);
    controller.abort();

    await assertion;
  });
});
