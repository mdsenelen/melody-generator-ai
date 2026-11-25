import { NextRequest, NextResponse } from "next/server";

const DEFAULT_BACKEND_BASE_URL = "http://127.0.0.1:8000";
const RESPONSE_HEADER_ALLOWLIST = [
  "cache-control",
  "content-disposition",
  "content-type",
  "etag",
  "last-modified",
  // retry-after: how long to wait before retrying a 429/503 (HEAVY_WORK_LOCK
  // busy, or the boot warm-up grace window). x-error-code: distinguishes
  // those two 429 cases from each other (both share the same status code).
  "retry-after",
  "x-error-code",
];

function isPlaceholderBackendUrl(value: string) {
  return /your-backend-url\.onrender\.com/i.test(value);
}

export function getBackendBaseUrl() {
  const configuredBaseUrl =
    process.env.BACKEND_BASE_URL ??
    process.env.BACKEND_URL ??
    process.env.NEXT_PUBLIC_BACKEND_URL ??
    DEFAULT_BACKEND_BASE_URL;
  const rawBaseUrl = isPlaceholderBackendUrl(configuredBaseUrl)
    ? DEFAULT_BACKEND_BASE_URL
    : configuredBaseUrl;

  const normalizedBaseUrl = rawBaseUrl.replace(/\/+$/, "");
  return normalizedBaseUrl.endsWith("/api") ? normalizedBaseUrl.slice(0, -4) : normalizedBaseUrl;
}

export function getBackendApiUrl(path: string) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${getBackendBaseUrl()}/api${normalizedPath}`;
}

async function buildProxyResponse(response: Response) {
  const headers = new Headers();

  for (const headerName of RESPONSE_HEADER_ALLOWLIST) {
    const value = response.headers.get(headerName);
    if (value) {
      headers.set(headerName, value);
    }
  }

  // Do not forward the upstream length: an intermediary may repackage the
  // body, leaving the browser with a truncated JSON response.
  const body = await response.arrayBuffer();

  return new NextResponse(body, {
    status: response.status,
    headers,
  });
}

// A backend that is down, restarting or still cold is a gateway problem, not a
// crash in this app. Answering 502 with a JSON detail keeps two things honest:
// the client's retry policy treats 502/503 as transient (src/domain/retryPolicy),
// so the UI says "waking up" and retries; and requestJson finds a `detail` to
// show instead of logging "[request] non-JSON error response". Letting the
// fetch throw would surface as an empty 500 with no content-type, which the
// policy classifies as permanent.
const BACKEND_UNAVAILABLE_DETAIL =
  "The server is unreachable or restarting. Please try again in a moment.";

function backendUnavailableResponse() {
  return new NextResponse(JSON.stringify({ detail: BACKEND_UNAVAILABLE_DETAIL }), {
    status: 502,
    headers: { "content-type": "application/json" },
  });
}

export async function proxyBackendGet(path: string, request?: NextRequest) {
  const headers = new Headers();
  const accept = request?.headers.get("accept");
  const backendApiUrl = getBackendApiUrl(path);

  if (accept) {
    headers.set("accept", accept);
  }

  let response: Response;

  try {
    response = await fetch(backendApiUrl, {
      method: "GET",
      headers,
      cache: "no-store",
      redirect: "manual",
    });
  } catch (error) {
    console.error("[backend proxy] fetch failed", { method: "GET", url: backendApiUrl, error });
    return backendUnavailableResponse();
  }

  return buildProxyResponse(response);
}

export async function proxyBackendRequest(request: NextRequest, path: string) {
  const backendApiUrl = getBackendApiUrl(path);
  const headers = new Headers();
  const accept = request.headers.get("accept");
  const contentType = request.headers.get("content-type") ?? "";
  let body: BodyInit | undefined;

  if (accept) {
    headers.set("accept", accept);
  }

  if (request.method !== "GET" && request.method !== "HEAD") {
    if (contentType.includes("multipart/form-data")) {
      body = await request.formData();
    } else {
      const rawBody = await request.text();
      body = rawBody;

      if (contentType) {
        headers.set("content-type", contentType);
      }
    }
  }

  console.log("[backend proxy] forwarding request", {
    method: request.method,
    url: backendApiUrl,
  });

  let response: Response;

  try {
    response = await fetch(backendApiUrl, {
      method: request.method,
      headers,
      body,
      cache: "no-store",
      redirect: "manual",
    });
  } catch (error) {
    console.error("[backend proxy] fetch failed", {
      method: request.method,
      url: backendApiUrl,
      error,
    });
    return backendUnavailableResponse();
  }

  console.log("[backend proxy] backend response", {
    method: request.method,
    url: backendApiUrl,
    status: response.status,
    ok: response.ok,
  });

  return buildProxyResponse(response);
}
