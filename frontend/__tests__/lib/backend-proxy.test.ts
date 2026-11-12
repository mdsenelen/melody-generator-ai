/**
 * @jest-environment node
 */
// NextRequest needs the web Request/Response globals, which this project's
// jsdom test environment doesn't provide -- this file exercises server-side
// route code, not DOM, so it belongs in the node environment regardless of
// living under __tests__/lib/ (jest.config.js's default for that directory).
import { NextRequest } from "next/server";

import { proxyBackendGet, proxyBackendRequest } from "../../app/api/_lib/backend";

describe("proxyBackendRequest response header allowlist", () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("forwards retry-after and x-error-code from the backend response", async () => {
    global.fetch = jest.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "The server is starting up." }), {
        status: 429,
        headers: {
          "content-type": "application/json",
          "retry-after": "60",
          "x-error-code": "warming_up",
        },
      }),
    ) as unknown as typeof fetch;

    const request = new NextRequest("http://localhost/api/generate-variants", { method: "POST" });
    const response = await proxyBackendRequest(request, "/generate-variants");

    expect(response.status).toBe(429);
    expect(response.headers.get("retry-after")).toBe("60");
    expect(response.headers.get("x-error-code")).toBe("warming_up");
  });

  it("does not forward headers outside the allowlist", async () => {
    global.fetch = jest.fn().mockResolvedValue(
      new Response("{}", {
        status: 200,
        headers: { "content-type": "application/json", "x-internal-debug": "secret" },
      }),
    ) as unknown as typeof fetch;

    const request = new NextRequest("http://localhost/api/generate-variants", { method: "POST" });
    const response = await proxyBackendRequest(request, "/generate-variants");

    expect(response.headers.get("x-internal-debug")).toBeNull();
  });
});

// An unreachable backend used to make these helpers throw, which Next turns into
// an empty 500 with no content-type: requestJson then had no `detail` to show,
// logged "[request] non-JSON error response", and the retry policy classified it
// permanent. Both helpers must answer 502 JSON instead. See backend.ts.
describe("proxy helpers when the backend is unreachable", () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  function failFetch() {
    global.fetch = jest
      .fn()
      .mockRejectedValue(new TypeError("fetch failed")) as unknown as typeof fetch;
    jest.spyOn(console, "error").mockImplementation(() => {});
  }

  it("proxyBackendGet answers 502 JSON instead of throwing", async () => {
    failFetch();

    const request = new NextRequest("http://localhost/api/progressions");
    const response = await proxyBackendGet("/progressions", request);

    expect(response.status).toBe(502);
    expect(response.headers.get("content-type")).toBe("application/json");
    await expect(response.json()).resolves.toEqual({ detail: expect.any(String) });
  });

  it("proxyBackendRequest answers 502 JSON instead of throwing", async () => {
    failFetch();

    const request = new NextRequest("http://localhost/api/generate-variants", { method: "POST" });
    const response = await proxyBackendRequest(request, "/generate-variants");

    expect(response.status).toBe(502);
    expect(response.headers.get("content-type")).toBe("application/json");
    await expect(response.json()).resolves.toEqual({ detail: expect.any(String) });
  });
});
