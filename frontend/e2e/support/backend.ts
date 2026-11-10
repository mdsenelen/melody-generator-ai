// Browser-level stand-in for FastAPI. Every request to the backend origin
// and to the Next.js /api/* proxy routes is answered here from
// e2e/fixtures/, and anything not declared fails loudly (501) instead of
// reaching a real server -- so no test ever runs inference.
import { readFileSync } from "node:fs";
import path from "node:path";

import type { Page, Route } from "@playwright/test";

const BACKEND = "http://127.0.0.1:8000";

function fixture<T>(name: string): T {
  return JSON.parse(readFileSync(path.join(__dirname, "..", "fixtures", name), "utf8")) as T;
}

export type Transcription = { midi_b64: string; midi_filename: string } & Record<string, unknown>;
export type Variants = {
  job_id: string;
  variants: { midi_b64: string; midi_filename: string }[];
} & Record<string, unknown>;

export const transcription = fixture<Transcription>("transcription.json");
export const analysis = fixture<Record<string, unknown>>("analysis.json");
export const variants = fixture<Variants>("variants.json");

export const UPLOAD = { id: "u1", filename: "upload_u1.wav" };
export const TRANSCRIBE_JOB_ID = "job-1";

export type JobState = "queued" | "processing" | "completed";

export function jobStatus(jobId: string, status: JobState, result: unknown = null) {
  return {
    job_id: jobId,
    status,
    progress: status === "completed" ? 1 : 0,
    result: status === "completed" ? result : null,
    error: null,
  };
}

type Handler = (route: Route) => Promise<void> | void;

export type BackendOptions = {
  /** Answers POST /api/generate-variants (the Next.js proxy route). */
  generateVariants?: Handler;
  /** What GET /api/transcribe/job-1 reports; read on every poll. */
  transcribeJob?: () => JobState;
};

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

export async function mockBackend(page: Page, options: BackendOptions = {}) {
  const transcribeJob = options.transcribeJob ?? (() => "completed" as const);

  // Registered first, so it only answers what nothing below declares.
  await page.route(`${BACKEND}/**`, (route) =>
    json(
      route,
      { detail: `e2e: unmocked ${route.request().method()} ${route.request().url()}` },
      501,
    ),
  );
  await page.route(`${BACKEND}/health`, (route) => json(route, { status: "ok" }));
  await page.route(`${BACKEND}/api/upload`, (route) => json(route, UPLOAD));
  await page.route(`${BACKEND}/api/transcribe`, (route) =>
    json(route, jobStatus(TRANSCRIBE_JOB_ID, "queued")),
  );
  await page.route(`${BACKEND}/api/transcribe/${TRANSCRIBE_JOB_ID}`, (route) =>
    json(route, jobStatus(TRANSCRIBE_JOB_ID, transcribeJob(), transcription)),
  );
  await page.route(`${BACKEND}/api/transcribe/${variants.job_id}`, (route) =>
    json(route, jobStatus(variants.job_id, "completed", variants)),
  );
  await page.route(`${BACKEND}/api/analyze`, (route) => json(route, analysis));

  // Same-origin Next.js proxy routes, answered before they reach the server.
  // (Downloads are navigations, which page.route can't see: those reach
  // e2e/support/stub-backend.mjs through the real /api/download route.)
  await page.route(
    "**/api/generate-variants",
    options.generateVariants ?? ((r) => json(r, variants)),
  );
}

/** A clip already analysed on /analyse, as the session store keeps it. */
export async function rememberAnalysedClip(page: Page) {
  await page.addInitScript(() => {
    window.sessionStorage.setItem(
      "melody-session",
      JSON.stringify({
        state: {
          lastUpload: {
            uploadId: "u1",
            filename: "upload_u1.wav",
            sourceName: "riff.wav",
            jobId: "job-1",
            transcription: {
              chords: ["C"],
              key: "C major",
              moodLabel: "happy",
              pitchHistogram: [],
            },
          },
        },
        version: 0,
      }),
    );
  });
}

/**
 * A .wav the browser can't read: its duration probe fails, so /analyse takes
 * the server path (mocked) instead of running Basic Pitch in the tab. Built
 * in memory -- no audio file is committed.
 */
export function unreadableWav(name = "riff.wav") {
  return { name, mimeType: "audio/wav", buffer: Buffer.from("RIFF\0\0\0\0WAVEnot-really-audio") };
}
