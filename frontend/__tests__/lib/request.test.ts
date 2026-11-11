import * as z from "zod/mini";

import { ContractError, requestJson, type RequestError } from "../../app/lib/request";
import { classifyFailure } from "../../src/domain/retryPolicy";

const anyBody = z.unknown();

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

    await expect(requestJson("/api/generate-variants", anyBody)).rejects.toMatchObject({
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
      await requestJson("/api/generate-variants", anyBody);
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

    await expect(requestJson("/api/upload", anyBody)).rejects.toBe(abortError);
  });

  it("replaces a network-level TypeError with a readable message", async () => {
    global.fetch = jest
      .fn()
      .mockRejectedValue(new TypeError("Failed to fetch")) as unknown as typeof fetch;

    await expect(requestJson("/api/upload", anyBody)).rejects.toThrow(/couldn't reach the server/i);
  });

  it("tags a network-level TypeError with code 'network_error' so callers can distinguish it from any other status-less error", async () => {
    global.fetch = jest
      .fn()
      .mockRejectedValue(new TypeError("Failed to fetch")) as unknown as typeof fetch;

    let caught: RequestError | undefined;
    try {
      await requestJson("/api/upload", anyBody);
    } catch (error) {
      caught = error as RequestError;
    }

    expect(caught?.code).toBe("network_error");
    expect(caught?.status).toBeUndefined();
  });
});

describe("requestJson parses the body at the boundary", () => {
  const schema = z.object({ id: z.string(), filename: z.string() });

  function respondWith(body: unknown) {
    jest.spyOn(global, "fetch").mockResolvedValue(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  }

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("returns the parsed value, without keys the schema doesn't know", async () => {
    respondWith({ id: "u1", filename: "upload_u1.wav", extra: true });

    await expect(requestJson("/api/upload", schema)).resolves.toEqual({
      id: "u1",
      filename: "upload_u1.wav",
    });
  });

  it("throws a non-retryable ContractError with a user-safe message when the body doesn't match", async () => {
    respondWith({ id: 42 });
    const consoleError = jest.spyOn(console, "error").mockImplementation(() => {});

    let caught: unknown;
    try {
      await requestJson("/api/upload", schema);
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(ContractError);
    expect(caught).toMatchObject({ name: "ContractError", code: "contract_error" });
    expect(classifyFailure(caught)).toBe("contract");
    // No zod internals in what the user reads...
    expect(caught).toHaveProperty(
      "message",
      expect.not.stringMatching(/expected|received|filename/i),
    );
    // ...but the issues are logged for whoever debugs it.
    expect(consoleError).toHaveBeenCalledWith(
      "[request] response failed its contract",
      expect.objectContaining({ url: "/api/upload", issues: expect.any(Array) }),
    );
  });
});
