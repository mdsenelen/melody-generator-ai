"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { MidiParseError, parseMidi, type PlayableNote } from "../app/lib/midi";

// Client-side playback of the base64 MIDI the generation endpoints return.
// Tone.js is loaded with a dynamic import() inside the play handler -- never at
// module scope -- so it stays out of SSR and the initial bundle, and the
// AudioContext is only created after a user gesture.

export type MidiPlayerState =
  | { status: "empty" }
  | { status: "idle"; durationSec: number }
  | { status: "loading"; durationSec: number }
  | { status: "playing"; durationSec: number; positionSec: number }
  | { status: "paused"; durationSec: number; positionSec: number }
  | { status: "error"; message: string };

export type MidiPlayer = {
  state: MidiPlayerState;
  toggle: () => void;
  stop: () => void;
};

const SAMPLE_BASE_URL = "/audio/piano/";
// Salamander Grand Piano, every tritone C2-C6 (Tone.Sampler pitch-shifts to
// fill the gaps). Keys are scientific pitch names; "#" -> "s" in the filename.
const SAMPLE_URLS: Record<string, string> = {
  C2: "C2.mp3",
  "F#2": "Fs2.mp3",
  C3: "C3.mp3",
  "F#3": "Fs3.mp3",
  C4: "C4.mp3",
  "F#4": "Fs4.mp3",
  C5: "C5.mp3",
  "F#5": "Fs5.mp3",
  C6: "C6.mp3",
};

const POSITION_POLL_MS = 200;

type ToneModule = typeof import("tone");

type Transport = {
  seconds: number;
  position: string | number;
  start: (time?: string | number) => unknown;
  pause: (time?: string | number) => unknown;
  stop: (time?: string | number) => unknown;
  cancel: (after?: number) => unknown;
};

type Sampler = { dispose: () => void; toDestination: () => unknown };
type Part = { dispose: () => void; start: (time: number) => unknown };
type Graph = { sampler: Sampler; part: Part; transport: Transport };

export function useMidiPlayer(midiB64: string | null): MidiPlayer {
  const parsed = useMemo(() => {
    if (!midiB64) return { ok: true as const, notes: [] as PlayableNote[], durationSec: 0 };
    try {
      const { notes, durationSec } = parseMidi(midiB64);
      return { ok: true as const, notes, durationSec };
    } catch (error) {
      return {
        ok: false as const,
        message:
          error instanceof MidiParseError
            ? "This melody couldn't be read for playback."
            : "This melody couldn't be prepared for playback.",
      };
    }
  }, [midiB64]);

  const hasNotes = parsed.ok && parsed.notes.length > 0;
  const durationSec = parsed.ok ? parsed.durationSec : 0;

  const initialState: MidiPlayerState = !parsed.ok
    ? { status: "error", message: parsed.message }
    : hasNotes
      ? { status: "idle", durationSec }
      : { status: "empty" };

  const [state, setState] = useState<MidiPlayerState>(initialState);

  const graphRef = useRef<Graph | null>(null);
  const toneRef = useRef<ToneModule | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Bumped whenever the source changes or the component unmounts, so an
  // in-flight sampler load knows it has been superseded and bails.
  const loadTokenRef = useRef(0);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const teardown = useCallback(() => {
    stopPolling();
    const graph = graphRef.current;
    if (graph) {
      try {
        graph.transport.stop();
        graph.transport.cancel(0);
        graph.part.dispose();
        graph.sampler.dispose();
      } catch {
        // disposing a half-built graph is best-effort
      }
      graphRef.current = null;
    }
  }, [stopPolling]);

  // Reset when the MIDI source changes (and tear down on unmount). initialState
  // is derived from midiB64, so keying the effect on midiB64 is enough.
  const initialStateRef = useRef(initialState);
  initialStateRef.current = initialState;
  useEffect(() => {
    loadTokenRef.current += 1;
    teardown();
    setState(initialStateRef.current);
    return () => {
      loadTokenRef.current += 1;
      teardown();
    };
  }, [midiB64, teardown]);

  const startPolling = useCallback(() => {
    stopPolling();
    pollRef.current = setInterval(() => {
      const graph = graphRef.current;
      if (!graph) return;
      const positionSec = graph.transport.seconds;
      if (durationSec > 0 && positionSec >= durationSec) {
        graph.transport.stop();
        graph.transport.position = 0;
        stopPolling();
        setState({ status: "paused", durationSec, positionSec: 0 });
        return;
      }
      setState({ status: "playing", durationSec, positionSec });
    }, POSITION_POLL_MS);
  }, [durationSec, stopPolling]);

  const startPlayback = useCallback(async () => {
    if (!parsed.ok || parsed.notes.length === 0) return;
    const token = loadTokenRef.current;
    setState({ status: "loading", durationSec });

    try {
      let Tone = toneRef.current;
      if (!Tone) {
        Tone = await import("tone");
        toneRef.current = Tone;
      }
      await Tone.start();
      if (token !== loadTokenRef.current) return;

      const sampler = await new Promise<Sampler>((resolve, reject) => {
        const s: Sampler = new Tone!.Sampler({
          urls: SAMPLE_URLS,
          baseUrl: SAMPLE_BASE_URL,
          onload: () => resolve(s),
          onerror: (err: Error) => reject(err),
        }) as unknown as Sampler;
      });
      if (token !== loadTokenRef.current) {
        sampler.dispose();
        return;
      }
      sampler.toDestination();

      const part = new Tone.Part(
        (time: number, note: PlayableNote) => {
          (
            sampler as unknown as {
              triggerAttackRelease: (n: string, d: number, t: number, v: number) => void;
            }
          ).triggerAttackRelease(note.name, note.duration, time, note.velocity);
        },
        parsed.notes.map((note) => [note.time, note] as [number, PlayableNote]),
      ) as unknown as Part;
      part.start(0);

      const transport = Tone.getTransport() as unknown as Transport;
      transport.position = 0;
      graphRef.current = { sampler, part, transport };
      transport.start();
      setState({ status: "playing", durationSec, positionSec: 0 });
      startPolling();
    } catch {
      if (token !== loadTokenRef.current) return;
      teardown();
      setState({ status: "error", message: "Playback isn't available in this browser." });
    }
  }, [parsed, durationSec, startPolling, teardown]);

  const toggle = useCallback(() => {
    if (state.status === "playing") {
      graphRef.current?.transport.pause();
      stopPolling();
      setState({ status: "paused", durationSec, positionSec: state.positionSec });
      return;
    }
    if (state.status === "paused") {
      graphRef.current?.transport.start();
      setState({ status: "playing", durationSec, positionSec: state.positionSec });
      startPolling();
      return;
    }
    if (state.status === "idle") {
      void startPlayback();
    }
  }, [state, durationSec, startPlayback, startPolling, stopPolling]);

  const stop = useCallback(() => {
    const graph = graphRef.current;
    stopPolling();
    if (graph) {
      graph.transport.stop();
      graph.transport.position = 0;
      setState({ status: "paused", durationSec, positionSec: 0 });
    } else if (hasNotes) {
      setState({ status: "idle", durationSec });
    }
  }, [durationSec, hasNotes, stopPolling]);

  return { state, toggle, stop };
}
