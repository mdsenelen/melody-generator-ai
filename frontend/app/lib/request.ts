type RequestJsonOptions = RequestInit & {
  expectedContentType?: string;
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
  const response = await fetch(input, options);
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
    const error = new Error(detail || `Request failed (${response.status})`) as Error & {
      status?: number;
    };
    error.status = response.status;
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
