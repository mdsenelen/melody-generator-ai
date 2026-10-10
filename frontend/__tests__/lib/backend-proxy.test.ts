/**
 * @jest-environment node
 */
// NextRequest needs the web Request/Response globals, which this project's
// jsdom test environment doesn't provide -- this file exercises server-side
// route code, not DOM, so it belongs in the node environment regardless of
// living under __tests__/lib/ (jest.config.js's default for that directory).
import { NextRequest } from "next/server";

import { proxyBackendRequest } from "../../app/api/_lib/backend";

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
