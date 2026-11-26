"use client";

import { useEffect, useState } from "react";

import { decodeToMono22050 } from "../app/lib/browserTranscribe/decode";
import { waveformPeaks } from "../src/domain/noteRoll";

const BUCKETS = 1600;

export type WaveformState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; peaks: Float32Array }
  | { status: "error" };

/**
 * Decodes the picked file (native, off-thread decodeAudioData) and reduces it
 * to a fixed-size envelope. The raw samples are dropped right after; only the
 * 1600-float envelope is kept. A failure only costs the waveform lane, never
 * the piano roll, so it is a state, not a thrown error.
 */
export function useWaveformPeaks(file: File | null): WaveformState {
  const [state, setState] = useState<{ file: File; value: WaveformState } | null>(null);

  useEffect(() => {
    if (!file) return;
    let cancelled = false; // decodeAudioData has no abort signal; drop stale results instead
    decodeToMono22050(file)
      .then(({ audio }) => {
        if (!cancelled) {
          setState({ file, value: { status: "ready", peaks: waveformPeaks(audio, BUCKETS) } });
        }
      })
      .catch(() => {
        if (!cancelled) setState({ file, value: { status: "error" } });
      });
    return () => {
      cancelled = true;
    };
  }, [file]);

  if (!file) return { status: "idle" };
  return state?.file === file ? state.value : { status: "loading" };
}
