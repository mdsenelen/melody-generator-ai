"use client";

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

export type TranscriptionSummary = {
  chords: string[];
  key: string;
  moodLabel: "happy" | "sad" | "neutral";
  pitchHistogram: number[];
};

export type UploadSession = {
  uploadId: string;
  filename: string;
  sourceName: string;
  transcription: TranscriptionSummary;
  // The completed transcribe job this upload produced, when there is one --
  // lets generate-variants skip re-transcribing the same audio (see
  // app/generate-variants/page.tsx). Optional: a direct upload on
  // generate-variants itself (no prior /analyse trip) has no job to point at.
  jobId?: string;
  // The clip window (seconds) last analysed, so coming back to /analyse
  // reopens on the same section instead of the default one.
  clipWindow?: { start: number; end: number };
};

type SessionState = {
  lastUpload: UploadSession | null;
  setLastUpload: (session: UploadSession) => void;
  clearLastUpload: () => void;
};

// sessionStorage doesn't exist during Next.js server rendering; fall back to
// a no-op storage there so the store can be imported by client components
// without throwing on the server render pass.
const noopStorage = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
};

export const useSessionStore = create<SessionState>()(
  persist(
    (set) => ({
      lastUpload: null,
      setLastUpload: (session) => set({ lastUpload: session }),
      clearLastUpload: () => set({ lastUpload: null }),
    }),
    {
      name: "melody-session",
      storage: createJSONStorage(() =>
        typeof window !== "undefined" ? window.sessionStorage : noopStorage,
      ),
    },
  ),
);
