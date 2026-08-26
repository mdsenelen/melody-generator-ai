"use client";

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
};

type ChordGraphProps = {
  title: string;
  description?: string;
  progression: string[];
  initialBpm?: number;
  initialInstrument?: number;
  genreBadge?: ReactNode;
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
    <section className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/20 backdrop-blur-md">
      <div className="flex flex-col gap-5">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2
              className="text-xl font-semibold text-white"
              style={{ textShadow: "0 2px 8px rgba(0,0,0,0.8)" }}
            >
              {title}
            </h2>
            {genreBadge ?? null}
          </div>
          {description ? <p className="mt-2 text-sm text-white/70">{description}</p> : null}
        </div>

        <div className="flex flex-wrap gap-2">
          {progression.map((chord, index) => (
            <div key={`${chord}-${index}`} className="flex items-center gap-2">
              <ChordDiagram chord={chord} />
              {index < progression.length - 1 ? <span className="text-white/45">→</span> : null}
            </div>
          ))}
        </div>

        <div className="grid gap-4 md:grid-cols-[1fr_auto]">
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
            <span className="font-medium text-white">Tempo: {bpm} BPM</span>
            <input
              type="range"
              min={60}
              max={160}
              step={1}
              value={bpm}
              onChange={(event) => setBpm(Number(event.target.value))}
              className="accent-purple-400"
            />
          </label>
          <button
            type="button"
            onClick={handleGenerate}
            disabled={loading}
            className="h-fit rounded-2xl border border-purple-400/40 bg-purple-500/15 px-4 py-3 text-sm font-semibold text-purple-100 transition hover:border-purple-300 hover:bg-purple-500/20 disabled:cursor-not-allowed disabled:opacity-60"
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
            <div className="mt-4 flex flex-wrap gap-3">
              <a
                href={result.midi_download_path ?? `data:audio/midi;base64,${result.midi_b64}`}
                download={result.midi_filename}
                className="rounded-full border border-sky-400/40 bg-sky-500/10 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:border-sky-300 hover:bg-sky-500/20"
              >
                Download MIDI
              </a>
              <a
                href={result.wav_download_path}
                download={result.wav_filename}
                className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:border-emerald-300 hover:bg-emerald-500/20"
              >
                Download WAV
              </a>
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}
