"use client";

import { useState } from "react";

import { MidiPlayer } from "../../../components/midi-player";
import { Spinner } from "../../../components/spinner";
import { Card } from "../../../components/ui/card";
import { Label } from "../../../components/ui/text";
import { VariantPicker } from "../../../components/ui/variant-picker";
import { useJobResult } from "../../../hooks/use-job-result";
import { isProgressionJobResult, isVariantsJobResult } from "../../lib/jobResult";

function downloadPathFor(filename: string) {
  return `/api/download/${encodeURIComponent(filename)}`;
}

const DOWNLOAD_MIDI_CLASS =
  "rounded-[var(--radius)] border border-primary/40 bg-primary/10 px-4 py-2 text-sm text-primary transition-colors hover:border-primary/70 hover:bg-primary/20";
const DOWNLOAD_WAV_CLASS =
  "rounded-[var(--radius)] border border-emerald-400/40 bg-emerald-500/10 px-4 py-2 text-sm text-emerald-100 transition-colors hover:border-emerald-300 hover:bg-emerald-500/20";

function Frame({ children }: { children: React.ReactNode }) {
  return (
    <Card>
      <div aria-live="polite">{children}</div>
    </Card>
  );
}

export default function ResultView({ jobId }: { jobId: string }) {
  const state = useJobResult(jobId);
  const [activeVariant, setActiveVariant] = useState(0);

  if (state.status === "loading" || state.status === "running") {
    return (
      <Frame>
        <div className="flex flex-col items-center justify-center gap-4 py-16">
          <Spinner
            size="lg"
            label={
              state.status === "running" && state.jobStatus === "processing"
                ? "Still working on your result..."
                : "Looking up your result..."
            }
          />
          <p className="text-muted-foreground max-w-xs text-center text-xs">
            This page updates automatically once your result is ready — no need to refresh.
          </p>
        </div>
      </Frame>
    );
  }

  if (state.status === "gone") {
    return (
      <Frame>
        <div className="border-border text-muted-foreground flex min-h-[240px] flex-col items-center justify-center gap-3 rounded-[var(--radius)] border border-dashed p-8 text-center">
          <p className="font-display text-foreground text-lg font-light">
            This result isn&apos;t available
          </p>
          <p className="max-w-md text-sm">
            The link may be mistyped, or the result has expired — generated files are only kept for
            a limited time. Try generating it again.
          </p>
        </div>
      </Frame>
    );
  }

  if (state.status === "failed") {
    return (
      <Frame>
        <div className="rounded-[var(--radius)] border border-red-500/40 bg-[#120808] p-4 text-sm text-red-100">
          {state.error}
        </div>
      </Frame>
    );
  }

  const { result } = state;

  if (isVariantsJobResult(result)) {
    const variant = result.variants[activeVariant] ?? result.variants[0];
    return (
      <Frame>
        <div className="space-y-6">
          <div>
            <Label>Result</Label>
            <h1 className="font-display text-foreground mt-1 text-2xl font-light">
              Generated melody variants
            </h1>
            <p className="text-muted-foreground mt-2 text-sm">Mood: {result.mood_label}</p>
          </div>

          <VariantPicker
            count={result.variants.length}
            active={activeVariant}
            onSelect={setActiveVariant}
          />

          {variant ? (
            <div className="border-border bg-background rounded-[var(--radius)] border p-5">
              <p className="text-muted-foreground text-sm">
                Temperature: <span className="text-foreground">{variant.temperature}</span>
              </p>
              <MidiPlayer midiB64={variant.midi_b64} className="mt-4" />
              <div className="mt-4 flex flex-wrap gap-3">
                <a
                  href={variant.midi_download_path || downloadPathFor(variant.midi_filename)}
                  download={variant.midi_filename}
                  className={DOWNLOAD_MIDI_CLASS}
                >
                  Download MIDI
                </a>
              </div>
            </div>
          ) : null}
        </div>
      </Frame>
    );
  }

  if (isProgressionJobResult(result)) {
    return (
      <Frame>
        <div className="space-y-6">
          <div>
            <Label>Result</Label>
            <h1 className="font-display text-foreground mt-1 text-2xl font-light">
              Rendered chord progression
            </h1>
            <p className="text-muted-foreground mt-2 text-sm">
              {result.progression.join(" → ")} · {result.bpm} BPM
            </p>
          </div>

          {result.wav_b64 ? (
            <audio controls className="w-full" src={`data:audio/wav;base64,${result.wav_b64}`}>
              Your browser does not support the audio element.
            </audio>
          ) : null}

          <div className="flex flex-wrap gap-3">
            <a
              href={result.midi_download_path || downloadPathFor(result.midi_filename)}
              download={result.midi_filename}
              className={DOWNLOAD_MIDI_CLASS}
            >
              Download MIDI
            </a>
            <a
              href={result.wav_download_path || downloadPathFor(result.wav_filename)}
              download={result.wav_filename}
              className={DOWNLOAD_WAV_CLASS}
            >
              Download WAV
            </a>
          </div>
        </div>
      </Frame>
    );
  }

  return (
    <Frame>
      <div className="space-y-6">
        <div>
          <Label>Result</Label>
          <h1 className="font-display text-foreground mt-1 text-2xl font-light">
            Transcription complete
          </h1>
          <p className="text-muted-foreground mt-2 text-sm">
            {result.n_notes} notes
            {result.source_duration_sec
              ? ` · ${Math.round(result.source_duration_sec)}s of audio`
              : null}
          </p>
        </div>

        {/* Full-length transcription is MIDI only -- mood/key/tempo/chords and
            a clip preview come from the analyse page (POST /api/analyze). */}
        <div className="flex flex-wrap gap-3">
          <a
            href={downloadPathFor(result.midi_filename)}
            download={result.midi_filename}
            className={DOWNLOAD_MIDI_CLASS}
          >
            Download MIDI
          </a>
        </div>
      </div>
    </Frame>
  );
}
