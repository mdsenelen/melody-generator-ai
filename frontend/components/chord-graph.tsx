"use client";

import Link from "next/link";
import { useEffect, useState, type ReactNode } from "react";

import { ChordDiagram } from "./chord-diagram";
import { Button, buttonClass } from "./ui/button";
import { Slider } from "./ui/slider";
import { useRenderProgression } from "../hooks/use-render-progression";

export const INSTRUMENT_OPTIONS = [
  { label: "Piano", value: 0 },
  { label: "Nylon Guitar", value: 24 },
  { label: "Acoustic Guitar", value: 25 },
  { label: "Electric Guitar", value: 30 },
];

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
  const { render, isRendering: loading, error, lastResult: result } = useRenderProgression();
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  // One object URL per rendered result, revoked when the next replaces it
  // or the graph unmounts.
  const audioB64 = result?.audio_b64;
  useEffect(() => {
    if (!audioB64) {
      setAudioUrl(null);
      return;
    }
    const url = createAudioObjectUrl(audioB64);
    setAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [audioB64]);

  const handleGenerate = () => render({ progression, bpm, instrument });

  return (
    <section className="border-border bg-card rounded-[var(--radius)] border p-5 sm:p-6">
      <div className="flex flex-col gap-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="font-display text-foreground text-xl font-light">{title}</h2>
              {genreBadge ?? null}
            </div>
            {description ? (
              <p className="text-muted-foreground mt-2 text-sm">{description}</p>
            ) : null}
            <p className="text-muted-foreground mt-3 max-w-md text-sm leading-6"></p>
          </div>
          <span className="text-muted-foreground hidden font-mono text-xs sm:block">
            {progression.length} chords
          </span>
        </div>

        <div>
          <p className="text-muted-foreground mb-3 font-mono text-[10px] tracking-widest uppercase">
            Chord sequence
          </p>
          <div className="flex flex-wrap items-center gap-2">
            {progression.map((chord, index) => (
              <div key={`${chord}-${index}`} className="flex items-center gap-2">
                <span className="border-border bg-secondary text-foreground rounded-[var(--radius)] border px-3 py-2 font-mono text-sm">
                  {chord}
                </span>
                {index < progression.length - 1 ? (
                  <span className="text-muted-foreground">→</span>
                ) : null}
              </div>
            ))}
          </div>
        </div>

        <div>
          <p className="text-muted-foreground mb-3 font-mono text-[10px] tracking-widest uppercase">
            Guitar diagrams
          </p>
          <div className="flex flex-wrap gap-2">
            {progression.map((chord, index) => (
              <ChordDiagram key={`${chord}-diagram-${index}`} chord={chord} />
            ))}
          </div>
        </div>

        <div className="border-border grid gap-4 border-t pt-5 md:grid-cols-[1fr_auto] md:items-end">
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
          <Slider label="Tempo" value={bpm} min={60} max={160} unit=" BPM" onChange={setBpm} />
          <Button onClick={handleGenerate} disabled={loading || !canPlay} loading={loading}>
            {loading ? "Rendering…" : "Play progression"}
          </Button>
        </div>

        {error ? (
          <div className="rounded-[var(--radius)] border border-red-500/40 bg-[#120808] p-3 text-sm text-red-100">
            {error}
          </div>
        ) : null}

        {result && audioUrl ? (
          <div className="border-border bg-background rounded-[var(--radius)] border p-4">
            <audio controls className="w-full" src={audioUrl}>
              Your browser does not support the audio element.
            </audio>
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
              <a
                href={result.midi_download_path ?? `data:audio/midi;base64,${result.midi_b64}`}
                download={result.midi_filename}
                aria-label="Download MIDI"
                className="text-primary hover:text-foreground font-mono text-sm transition-colors"
              >
                ↓ MIDI
              </a>
              <Link href={`/result/${result.job_id}`} className={buttonClass("secondary")}>
                View &amp; download result
              </Link>
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}
