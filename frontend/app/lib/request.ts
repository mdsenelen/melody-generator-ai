type RequestJsonOptions = RequestInit & {
  expectedContentType?: string;
};

// Thrown by requestJson on a non-ok response. `code` and `retryAfterSeconds`
// are read from response headers (x-error-code, retry-after) rather than
// the JSON body, so `detail` stays a plain string everywhere -- no change
// to the shape of any existing error response.
export type RequestError = Error & {
  status?: number;
  code?: string;
  retryAfterSeconds?: number;
};

function isJsonContentType(contentType: string) {
  return contentType.toLowerCase().includes("application/json");
}

function tryParseJson<T>(raw: string): T | null {
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export async function requestJson<T>(input: RequestInfo | URL, options: RequestJsonOptions = {}) {
  let response: Response;
  try {
    response = await fetch(input, options);
  } catch (error) {
    // Browsers reject fetch with a TypeError ("Failed to fetch") on network
    // or CORS failures. Anything else (e.g. an AbortError) is the caller's.
    if (!(error instanceof TypeError)) {
      throw error;
    }
    // Say what the user can actually check instead of the opaque message.
    throw new Error(
      "Couldn't reach the server. Check your connection and try again; if it keeps happening the server may be down or waking up.",
    );
  }
  const contentType = response.headers.get("content-type") ?? "";
  const rawBody = await response.text();
  const parsedBody = isJsonContentType(contentType)
    ? (tryParseJson<T & { detail?: string }>(rawBody) ?? null)
    : (tryParseJson<T & { detail?: string }>(rawBody) ?? null);

  if (!response.ok) {
    const detail =
      parsedBody && typeof parsedBody === "object" && "detail" in parsedBody
        ? parsedBody.detail
        : null;
    if (!detail) {
      console.error("[request] non-JSON error response", {
        status: response.status,
        contentType,
        rawBody,
      });
    }
    const error = new Error(detail || `Request failed (${response.status})`) as RequestError;
    error.status = response.status;
    const code = response.headers.get("x-error-code");
    if (code) {
      error.code = code;
    }
    const retryAfterRaw = response.headers.get("retry-after");
    const retryAfterSeconds = retryAfterRaw ? Number(retryAfterRaw) : NaN;
    if (Number.isFinite(retryAfterSeconds)) {
      error.retryAfterSeconds = retryAfterSeconds;
    }
    throw error;
  }

  if (parsedBody === null) {
    const fallback = options.expectedContentType ?? "application/json";
    console.error("[request] unexpected response content type", {
      expected: fallback,
      contentType,
      rawBody,
    });
    throw new Error(`Expected ${fallback} response but received content-type "${contentType}"`);
  }

  return parsedBody as T;
}
