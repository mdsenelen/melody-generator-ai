"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { PitchDetector } from "pitchy";
import {
  frequencyToMidi,
  midiToPitchClass,
  midiToNoteName,
  isInPlayableRange,
  CLARITY_THRESHOLD,
} from "../utils/pitch";

export type AudioAnalyzerState = {
  currentNote: string | null; // e.g. "A4"
  currentFrequency: number | null; // Hz
  pitchClass: number | null; // 0–11 (index into NOTE_NAMES)
  clarity: number; // 0–1 confidence from Pitchy
  noteHistogram: number[]; // length 12, accumulated count per pitch class
};

const INITIAL_STATE: AudioAnalyzerState = {
  currentNote: null,
  currentFrequency: null,
  pitchClass: null,
  clarity: 0,
  noteHistogram: new Array(12).fill(0),
};

export function useAudioAnalyzer(stream: MediaStream | null, enabled: boolean): AudioAnalyzerState {
  const [state, setState] = useState<AudioAnalyzerState>(INITIAL_STATE);

  // Refs hold the Web Audio graph nodes so we can tear them down cleanly
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  // Pitchy detector instance — created once per stream, reused every frame
  const detectorRef = useRef<PitchDetector<Float32Array> | null>(null);
  // Buffer reused every frame to avoid per-frame allocations
  const bufferRef = useRef<Float32Array<ArrayBuffer> | null>(null);

  const stopAnalysis = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    analyserRef.current?.disconnect();
    analyserRef.current = null;
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    detectorRef.current = null;
    bufferRef.current = null;
  }, []);

  useEffect(() => {
    if (!enabled || !stream) {
      stopAnalysis();
      setState(INITIAL_STATE);
      return;
    }

    // --- Build the Web Audio graph ---
    const audioCtx = new AudioContext();
    const analyser = audioCtx.createAnalyser();

    // fftSize must be a power of 2. 2048 gives good frequency resolution (~21 Hz per bin at 44.1 kHz)
    // while still being fast enough for real-time use.
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0; // no smoothing — we want raw frames for pitch detection

    const source = audioCtx.createMediaStreamSource(stream);
    source.connect(analyser);
    // Deliberately NOT connecting to audioCtx.destination — avoids mic feedback

    audioCtxRef.current = audioCtx;
    analyserRef.current = analyser;

    // Create Pitchy detector sized to match the analyser buffer
    detectorRef.current = PitchDetector.forFloat32Array(analyser.fftSize);
    bufferRef.current = new Float32Array(
      new ArrayBuffer(analyser.fftSize * Float32Array.BYTES_PER_ELEMENT),
    );

    // Histogram accumulated across this recording session
    const histogram = new Array<number>(12).fill(0);

    // --- rAF loop: runs every browser frame (~16 ms at 60fps) ---
    let frameCount = 0;
    const loop = () => {
      // Skip analysis on odd frames — reduces Pitchy work to ~30fps while rAF stays at 60fps
      if (++frameCount % 2 !== 0) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      const analyserNode = analyserRef.current;
      const detector = detectorRef.current;
      const buffer = bufferRef.current;

      if (!analyserNode || !detector || !buffer) return;

      // Fill buffer with time-domain audio samples (range -1 to +1)
      analyserNode.getFloatTimeDomainData(buffer);

      // Ask Pitchy for the best pitch estimate
      const [frequency, clarity] = detector.findPitch(buffer, audioCtx.sampleRate);

      if (clarity >= CLARITY_THRESHOLD && isInPlayableRange(frequency)) {
        const midi = frequencyToMidi(frequency);
        const pc = midiToPitchClass(midi);
        const noteName = midiToNoteName(midi);

        histogram[pc] += 1;

        setState({
          currentNote: noteName,
          currentFrequency: Math.round(frequency),
          pitchClass: pc,
          clarity,
          // Spread to create a new array reference so React re-renders
          noteHistogram: [...histogram],
        });
      } else {
        // Signal present but not pitched (noise, silence, breath) — keep histogram, clear current note
        setState((prev) => ({
          ...prev,
          currentNote: null,
          currentFrequency: null,
          pitchClass: null,
          clarity,
        }));
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);

    return () => {
      stopAnalysis();
    };
  }, [stream, enabled, stopAnalysis]);

  return state;
}
