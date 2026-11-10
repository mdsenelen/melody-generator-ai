import { safeParse } from "zod/mini";
import type { $ZodIssue, $ZodType, output } from "zod/v4/core";

import { HUNG_ATTEMPT_CODE } from "../../src/domain/retryPolicy";

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

function createRequestError(
  message: string,
  fields: { status?: number; code?: string; retryAfterSeconds?: number } = {},
): RequestError {
  return Object.assign(new Error(message), fields);
}

// Say what the user can actually check instead of the opaque "Failed to
// fetch". Tagged with a distinct code (rather than left as a bare Error) so
// callers can tell "couldn't reach the server at all" apart from any other
// status-less Error thrown elsewhere in their own request flow.
export function createNetworkError(
  message = "Couldn't reach the server. Check your connection and try again; if it keeps happening the server may be down or waking up.",
): RequestError {
  return createRequestError(message, { code: "network_error" });
}

// An attempt abandoned after its own timeout (see withAttemptTimeout): the
// server never answered, which on Render's free tier means it's still waking.
export function createHungAttemptError(): RequestError {
  return createRequestError("The server took too long to answer.", { code: HUNG_ATTEMPT_CODE });
}

// The response arrived but doesn't match its contract (src/domain/contracts).
// Retrying gets the same body back, so this is never retried; the zod issues
// go to the console, not to the user.
export class ContractError extends Error {
  readonly code = "contract_error";
  readonly issues: readonly $ZodIssue[];

  constructor(issues: readonly $ZodIssue[]) {
    super(
      "The server sent a response this page can't read. Please reload and try again; if it keeps happening, report it.",
    );
    this.name = "ContractError";
    this.issues = issues;
  }
}

function tryParseJson(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    return undefined;
  }
}

function detailOf(body: unknown): string | null {
  if (typeof body === "object" && body !== null && "detail" in body) {
    const { detail } = body;
    return typeof detail === "string" && detail ? detail : null;
  }
  return null;
}

function describeInput(input: RequestInfo | URL): string {
  if (typeof input === "string") return input;
  return input instanceof URL ? input.href : input.url;
}

/**
 * fetch + parse. The body is validated against `schema` and the parsed value
 * returned, so nothing downstream ever trusts an unchecked cast.
 */
export async function requestJson<S extends $ZodType>(
  input: RequestInfo | URL,
  schema: S,
  options: RequestJsonOptions = {},
): Promise<output<S>> {
  let response: Response;
  try {
    response = await fetch(input, options);
  } catch (error) {
    // Browsers reject fetch with a TypeError ("Failed to fetch") on network
    // or CORS failures. Anything else (e.g. an AbortError) is the caller's.
    if (!(error instanceof TypeError)) {
      throw error;
    }
    throw createNetworkError();
  }
  const contentType = response.headers.get("content-type") ?? "";
  const rawBody = await response.text();
  const body = tryParseJson(rawBody);

  if (!response.ok) {
    const detail = detailOf(body);
    if (!detail) {
      console.error("[request] non-JSON error response", {
        status: response.status,
        contentType,
        rawBody,
      });
    }
    const code = response.headers.get("x-error-code");
    const retryAfterRaw = response.headers.get("retry-after");
    const retryAfterSeconds = retryAfterRaw ? Number(retryAfterRaw) : NaN;
    throw createRequestError(detail || `Request failed (${response.status})`, {
      status: response.status,
      ...(code ? { code } : {}),
      ...(Number.isFinite(retryAfterSeconds) ? { retryAfterSeconds } : {}),
    });
  }

  if (body === undefined) {
    const fallback = options.expectedContentType ?? "application/json";
    console.error("[request] unexpected response content type", {
      expected: fallback,
      contentType,
      rawBody,
    });
    throw new Error(`Expected ${fallback} response but received content-type "${contentType}"`);
  }

  const parsed = safeParse(schema, body);
  if (!parsed.success) {
    console.error("[request] response failed its contract", {
      url: describeInput(input),
      issues: parsed.error.issues,
    });
    throw new ContractError(parsed.error.issues);
  }
  return parsed.data;
}
