import { requestJson, type RequestError } from "../../app/lib/request";

describe("requestJson error metadata", () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("surfaces status, x-error-code, and retry-after onto the thrown error", async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 429,
      headers: new Headers({
        "content-type": "application/json",
        "x-error-code": "warming_up",
        "retry-after": "60",
      }),
      text: async () => JSON.stringify({ detail: "The server is starting up." }),
    }) as unknown as typeof fetch;

    await expect(requestJson("/api/generate-variants")).rejects.toMatchObject({
      message: "The server is starting up.",
      status: 429,
      code: "warming_up",
      retryAfterSeconds: 60,
    } satisfies Partial<RequestError>);
  });

  it("leaves code and retryAfterSeconds undefined when the headers are absent", async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 500,
      headers: new Headers({ "content-type": "application/json" }),
      text: async () => JSON.stringify({ detail: "Something broke" }),
    }) as unknown as typeof fetch;

    let caught: RequestError | undefined;
    try {
      await requestJson("/api/generate-variants");
    } catch (error) {
      caught = error as RequestError;
    }

    expect(caught?.status).toBe(500);
    expect(caught?.code).toBeUndefined();
    expect(caught?.retryAfterSeconds).toBeUndefined();
  });

  it("passes an AbortError through unchanged instead of the network message", async () => {
    const abortError = new DOMException("The operation was aborted.", "AbortError");
    global.fetch = jest.fn().mockRejectedValue(abortError) as unknown as typeof fetch;

    await expect(requestJson("/api/upload")).rejects.toBe(abortError);
  });

  it("replaces a network-level TypeError with a readable message", async () => {
    global.fetch = jest
      .fn()
      .mockRejectedValue(new TypeError("Failed to fetch")) as unknown as typeof fetch;

    await expect(requestJson("/api/upload")).rejects.toThrow(/couldn't reach the server/i);
  });
});
