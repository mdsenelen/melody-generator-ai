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
