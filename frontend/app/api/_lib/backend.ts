import { NextRequest, NextResponse } from "next/server";

const DEFAULT_BACKEND_BASE_URL = "http://127.0.0.1:8000";
const RESPONSE_HEADER_ALLOWLIST = [
  "cache-control",
  "content-disposition",
  "content-length",
  "content-type",
  "etag",
  "last-modified",
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
  return normalizedBaseUrl.endsWith("/api")
    ? normalizedBaseUrl.slice(0, -4)
    : normalizedBaseUrl;
}

export function getBackendApiUrl(path: string) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${getBackendBaseUrl()}/api${normalizedPath}`;
}

function buildProxyResponse(response: Response) {
  const headers = new Headers();

  for (const headerName of RESPONSE_HEADER_ALLOWLIST) {
    const value = response.headers.get(headerName);
    if (value) {
      headers.set(headerName, value);
    }
  }

  return new NextResponse(response.body, {
    status: response.status,
    headers,
  });
}

export async function proxyBackendGet(path: string, request?: NextRequest) {
  const headers = new Headers();
  const accept = request?.headers.get("accept");

  if (accept) {
    headers.set("accept", accept);
  }

  const response = await fetch(getBackendApiUrl(path), {
    method: "GET",
    headers,
    cache: "no-store",
    redirect: "manual",
  });

  return buildProxyResponse(response);
}

export async function proxyBackendRequest(
  request: NextRequest,
  path: string,
) {
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
    throw error;
  }

  console.log("[backend proxy] backend response", {
    method: request.method,
    url: backendApiUrl,
    status: response.status,
    ok: response.ok,
  });

  return buildProxyResponse(response);
}
