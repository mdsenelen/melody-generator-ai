"use client";

import Link from "next/link";
import { useEffect, useState, type ReactNode } from "react";

import { ChordDiagram } from "./chord-diagram";
import { Spinner } from "./spinner";
import { requestJson } from "../app/lib/request";

export const INSTRUMENT_OPTIONS = [
  { label: "Piano", value: 0 },
  { label: "Nylon Guitar", value: 24 },
  { label: "Acoustic Guitar", value: 25 },
  { label: "Electric Guitar", value: 30 },
];

type ProgressionResponse = {
  audio_b64: string;
  midi_b64: string;
  midi_filename: string;
  midi_download_path: string;
  wav_filename: string;
  wav_download_path: string;
  bpm: number;
  instrument: number;
  job_id: string;
};

type ChordGraphProps = {
  title: string;
  description?: string;
  progression: string[];
  initialBpm?: number;
  initialInstrument?: number;
  genreBadge?: ReactNode;
  canPlay?: boolean;
};

function createAudioObjectUrl(base64Audio: string) {
  const bytes = Uint8Array.from(atob(base64Audio), (character) => character.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
}

export function ChordGraph({
  title,
  description,
  progression,
  initialBpm = 120,
  initialInstrument = 0,
  genreBadge,
  canPlay = true,
}: ChordGraphProps) {
  const [bpm, setBpm] = useState(initialBpm);
  const [instrument, setInstrument] = useState(initialInstrument);
  const [result, setResult] = useState<ProgressionResponse | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await requestJson<ProgressionResponse>("/api/generate-progression", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ progression, bpm, instrument }),
        expectedContentType: "application/json",
      });
      if (!data.audio_b64) {
        throw new Error("Could not render progression");
      }

      const nextAudioUrl = createAudioObjectUrl(data.audio_b64);
      setAudioUrl((current) => {
        if (current) {
          URL.revokeObjectURL(current);
        }
        return nextAudioUrl;
      });
      setResult(data);
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : "Could not render progression",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="rounded-[1.35rem] border border-white/10 bg-[rgba(17,22,32,0.76)] p-5 shadow-[0_12px_28px_rgba(2,6,23,0.2)] backdrop-blur-md sm:p-6">
      <div className="flex flex-col gap-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2
                className="text-xl font-semibold tracking-[-0.02em] text-white"
                style={{ textShadow: "0 2px 8px rgba(0,0,0,0.8)" }}
              >
                {title}
              </h2>
              {genreBadge ?? null}
            </div>
            {description ? <p className="mt-2 text-sm text-white/55">{description}</p> : null}
            <p className="mt-3 max-w-md text-sm leading-6 text-white/45">
              A versatile chord movement for building melodies, hooks, and richer harmonic ideas.
            </p>
          </div>
          <span className="hidden text-xs text-white/30 sm:block">{progression.length} chords</span>
        </div>

        <div>
          <p className="mb-3 text-[10px] font-semibold tracking-[0.2em] text-white/40 uppercase">
            Chord sequence
          </p>
          <div className="flex flex-wrap items-center gap-2">
            {progression.map((chord, index) => (
              <div key={`${chord}-${index}`} className="flex items-center gap-2">
                <span className="rounded-lg border border-white/10 bg-white/[0.06] px-3 py-2 text-sm font-semibold text-white">
                  {chord}
                </span>
                {index < progression.length - 1 ? <span className="text-white/25">→</span> : null}
              </div>
            ))}
          </div>
        </div>

        <div>
          <p className="mb-3 text-[10px] font-semibold tracking-[0.2em] text-white/40 uppercase">
            Guitar diagrams
          </p>
          <div className="flex flex-wrap gap-2">
            {progression.map((chord, index) => (
              <ChordDiagram key={`${chord}-diagram-${index}`} chord={chord} />
            ))}
          </div>
        </div>

        <div className="grid gap-4 border-t border-white/10 pt-5 md:grid-cols-[1fr_auto] md:items-end">
          {/* Instrument dropdown hidden — backend support coming later; defaults to Piano (0) */}
          {/*
          <label className="flex flex-col gap-2 text-sm text-white/75">
            <span className="font-medium text-white">Instrument</span>
            <select
              value={instrument}
              onChange={(event) => setInstrument(Number(event.target.value))}
              className="rounded-2xl border border-white/10 bg-black/20 px-3 py-2 text-white outline-none transition focus:border-purple-400"
            >
              {INSTRUMENT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          */}
          <label className="flex flex-col gap-2 text-sm text-white/75">
            <span className="flex items-center justify-between font-medium text-white">
              <span className="text-[10px] tracking-[0.2em] text-white/40 uppercase">Tempo</span>
              <span>{bpm} BPM</span>
            </span>
            <input
              type="range"
              min={60}
              max={160}
              step={1}
              value={bpm}
              onChange={(event) => setBpm(Number(event.target.value))}
              className="h-1 w-full cursor-pointer accent-[#a879ff]"
            />
          </label>
          <button
            type="button"
            onClick={handleGenerate}
            disabled={loading || !canPlay}
            className="h-fit rounded-xl border border-[#a879ff]/60 bg-[#8b5cf6]/25 px-5 py-3 text-sm font-semibold text-white transition hover:border-[#c09aff] hover:bg-[#8b5cf6]/35 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? <Spinner size="sm" label="Rendering" /> : "Play progression"}
          </button>
        </div>

        {error ? (
          <div className="rounded-2xl border border-red-500/40 bg-red-950/40 p-3 text-sm text-red-100">
            {error}
          </div>
        ) : null}

        {result && audioUrl ? (
          <div className="rounded-2xl border border-white/10 bg-black/20 p-4 backdrop-blur-sm">
            <audio controls className="w-full" src={audioUrl}>
              Your browser does not support the audio element.
            </audio>
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
              <a
                href={result.midi_download_path ?? `data:audio/midi;base64,${result.midi_b64}`}
                download={result.midi_filename}
                aria-label="Download MIDI"
                className="text-sm font-semibold text-[#c9a9ff] transition hover:text-white"
              >
                ↓ MIDI
              </a>
              <Link
                href={`/result/${result.job_id}`}
                className="rounded-full border border-sky-400/40 bg-sky-500/10 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:border-sky-300 hover:bg-sky-500/20"
              >
                View &amp; download result
              </Link>
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}
