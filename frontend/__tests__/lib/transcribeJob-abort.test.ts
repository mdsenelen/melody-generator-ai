import { http, HttpResponse } from "msw";

import { getTranscribeJob, pollTranscribeJob } from "../../app/lib/transcribeJob";
import { isCancellation } from "../../src/domain/errors";
import { BACKEND_API, server } from "../../src/test/msw/server";

const JOB_URL = `${BACKEND_API}/transcribe/job-1`;

function queued() {
  return HttpResponse.json({
    job_id: "job-1",
    status: "queued",
    progress: 0,
    result: null,
    error: null,
    waiting_reason: null,
  });
}

async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected the promise to reject");
}

describe("getTranscribeJob cancellation", () => {
  it("hands the caller's signal to the request", async () => {
    let requestSignal: AbortSignal | undefined;
    let requestSeen!: () => void;
    const seen = new Promise<void>((resolve) => (requestSeen = resolve));
    server.use(
      http.get(JOB_URL, ({ request }) => {
        requestSignal = request.signal;
        requestSeen();
        return new Promise<Response>(() => {}); // never answers on its own
      }),
    );

    const controller = new AbortController();
    const pending = getTranscribeJob("job-1", { signal: controller.signal });
    await seen;
    controller.abort();

    expect(isCancellation(await rejection(pending))).toBe(true);
    expect(requestSignal?.aborted).toBe(true);
  });
});

describe("pollTranscribeJob cancellation", () => {
  it("aborts the in-flight status request and rejects with a cancellation", async () => {
    let requestSignal: AbortSignal | undefined;
    let requestSeen!: () => void;
    const seen = new Promise<void>((resolve) => (requestSeen = resolve));
    server.use(
      http.get(JOB_URL, ({ request }) => {
        requestSignal = request.signal;
        requestSeen();
        return new Promise<Response>(() => {});
      }),
    );

    const controller = new AbortController();
    const polling = pollTranscribeJob("job-1", { signal: controller.signal });
    await seen;
    controller.abort();

    expect(isCancellation(await rejection(polling))).toBe(true);
    expect(requestSignal?.aborted).toBe(true);
  });

  it("rejects at once when aborted during the wait between polls, without polling again", async () => {
    let calls = 0;
    let firstAnswered!: () => void;
    const answered = new Promise<void>((resolve) => (firstAnswered = resolve));
    server.use(
      http.get(JOB_URL, () => {
        calls += 1;
        firstAnswered();
        return queued();
      }),
    );

    const controller = new AbortController();
    const onStatusChange = jest.fn();
    const polling = pollTranscribeJob("job-1", { signal: controller.signal, onStatusChange });
    await answered;
    // The first status lands, then the poll sleeps for a second. Aborting
    // must not wait that second out.
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(onStatusChange).toHaveBeenCalledTimes(1);

    const startedAt = Date.now();
    controller.abort();
    const error = await rejection(polling);

    expect(isCancellation(error)).toBe(true);
    expect(Date.now() - startedAt).toBeLessThan(500);
    expect(calls).toBe(1);
  });

  it("rejects without a request when the signal is already aborted", async () => {
    let calls = 0;
    server.use(
      http.get(JOB_URL, () => {
        calls += 1;
        return queued();
      }),
    );
    const controller = new AbortController();
    controller.abort();

    expect(
      isCancellation(await rejection(pollTranscribeJob("job-1", { signal: controller.signal }))),
    ).toBe(true);
    expect(calls).toBe(0);
  });
});
