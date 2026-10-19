import {
  explainUploadNetworkError,
  REACHABILITY_PROBE_TIMEOUT_MS,
  SERVER_REJECTED_MESSAGE,
  SERVER_UNREACHABLE_MESSAGE,
} from "../../app/lib/explainUploadNetworkError";
import { createNetworkError } from "../../app/lib/request";
import { isWakeupSignature, RetryCancelledError } from "../../app/lib/retryOnWakeup";

describe("explainUploadNetworkError", () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    jest.useRealTimers();
    global.fetch = originalFetch;
  });

  it("passes a non-network error through untouched, without probing", async () => {
    const fetchMock = jest.fn();
    global.fetch = fetchMock;
    const error = Object.assign(new Error("Audio is too long"), { status: 400 });

    await expect(explainUploadNetworkError(error)).resolves.toBe(error);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("says the server rejected the upload when an opaque no-cors probe still gets through", async () => {
    const fetchMock = jest.fn().mockResolvedValue({ type: "opaque", status: 0 } as Response);
    global.fetch = fetchMock;

    const explained = await explainUploadNetworkError(createNetworkError());

    expect(explained).toMatchObject({ code: "network_error", message: SERVER_REJECTED_MESSAGE });
    expect(isWakeupSignature(explained)).toBe(true);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/health$/);
    expect(String(url)).not.toMatch(/\/api\/health$/);
    expect(init).toMatchObject({ mode: "no-cors" });
  });

  it("says the server is restarting or unreachable when the probe can't connect either", async () => {
    global.fetch = jest.fn().mockRejectedValue(new TypeError("Failed to fetch"));

    const explained = await explainUploadNetworkError(createNetworkError());

    expect(explained).toMatchObject({ code: "network_error", message: SERVER_UNREACHABLE_MESSAGE });
  });

  it("treats a probe that hangs past its timeout as unreachable", async () => {
    jest.useFakeTimers();
    global.fetch = jest.fn().mockImplementation(
      (_input: RequestInfo | URL, init?: RequestInit) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          );
        }),
    );

    const promise = explainUploadNetworkError(createNetworkError());
    await jest.advanceTimersByTimeAsync(REACHABILITY_PROBE_TIMEOUT_MS);

    await expect(promise).resolves.toMatchObject({ message: SERVER_UNREACHABLE_MESSAGE });
  });

  it("rejects with RetryCancelledError when the caller aborts mid-probe", async () => {
    global.fetch = jest.fn().mockImplementation(
      (_input: RequestInfo | URL, init?: RequestInit) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          );
        }),
    );
    const controller = new AbortController();

    const promise = explainUploadNetworkError(createNetworkError(), controller.signal);
    controller.abort();

    await expect(promise).rejects.toBeInstanceOf(RetryCancelledError);
  });
});
