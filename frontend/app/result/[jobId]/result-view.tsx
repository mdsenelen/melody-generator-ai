"use client";

import { useState } from "react";

import { Spinner } from "../../../components/spinner";
import { useJobResult } from "../../../hooks/use-job-result";
import { isProgressionJobResult, isVariantsJobResult } from "../../lib/jobResult";

function downloadPathFor(filename: string) {
  return `/api/download/${encodeURIComponent(filename)}`;
}

function Frame({ children }: { children: React.ReactNode }) {
  return (
    <section className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/20 backdrop-blur-md">
      <div aria-live="polite">{children}</div>
    </section>
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
          <p className="max-w-xs text-center text-xs text-white/45">
            This page updates automatically once your result is ready — no need to refresh.
          </p>
        </div>
      </Frame>
    );
  }

  if (state.status === "gone") {
    return (
      <Frame>
        <div className="flex min-h-[240px] flex-col items-center justify-center gap-3 rounded-3xl border border-dashed border-white/15 bg-white/5 p-8 text-center text-white/65">
          <p className="text-lg font-semibold text-white">This result isn&apos;t available</p>
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
        <div className="rounded-2xl border border-red-500/40 bg-red-950/40 p-4 text-sm text-red-100">
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
            <p className="text-sm font-semibold text-white/75">Result</p>
            <h1 className="mt-1 text-2xl font-semibold text-white">Generated melody variants</h1>
            <p className="mt-2 text-sm text-white/65">Mood: {result.mood_label}</p>
          </div>

          <div className="flex flex-wrap gap-2">
            {result.variants.map((entry, index) => (
              <button
                key={entry.index}
                type="button"
                onClick={() => setActiveVariant(index)}
                className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                  activeVariant === index
                    ? "border border-purple-400/60 bg-purple-500/20 text-white"
                    : "border border-white/10 bg-white/5 text-white/70 hover:border-white/20 hover:text-white"
                }`}
              >
                Variant {index + 1}
              </button>
            ))}
          </div>

          {variant ? (
            <div className="rounded-3xl border border-white/10 bg-black/20 p-5">
              <p className="text-sm text-white/65">
                Temperature: <span className="font-semibold text-white">{variant.temperature}</span>
              </p>
              {variant.wav_b64 ? (
                <audio
                  controls
                  className="mt-4 w-full"
                  src={`data:audio/wav;base64,${variant.wav_b64}`}
                >
                  Your browser does not support the audio element.
                </audio>
              ) : null}
              <div className="mt-4 flex flex-wrap gap-3">
                <a
                  href={variant.midi_download_path || downloadPathFor(variant.midi_filename)}
                  download={variant.midi_filename}
                  className="rounded-full border border-sky-400/40 bg-sky-500/10 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:border-sky-300 hover:bg-sky-500/20"
                >
                  Download MIDI
                </a>
                {variant.wav_filename ? (
                  <a
                    href={variant.wav_download_path || downloadPathFor(variant.wav_filename)}
                    download={variant.wav_filename}
                    className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:border-emerald-300 hover:bg-emerald-500/20"
                  >
                    Download WAV
                  </a>
                ) : null}
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
            <p className="text-sm font-semibold text-white/75">Result</p>
            <h1 className="mt-1 text-2xl font-semibold text-white">Rendered chord progression</h1>
            <p className="mt-2 text-sm text-white/65">
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
              className="rounded-full border border-sky-400/40 bg-sky-500/10 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:border-sky-300 hover:bg-sky-500/20"
            >
              Download MIDI
            </a>
            <a
              href={result.wav_download_path || downloadPathFor(result.wav_filename)}
              download={result.wav_filename}
              className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:border-emerald-300 hover:bg-emerald-500/20"
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
          <p className="text-sm font-semibold text-white/75">Result</p>
          <h1 className="mt-1 text-2xl font-semibold text-white">Transcription complete</h1>
          <p className="mt-2 text-sm text-white/65">
            Key: {result.key} · Mood: {result.mood_label} · {result.tempo_bpm} BPM
          </p>
        </div>

        {result.wav_b64 ? (
          <audio controls className="w-full" src={`data:audio/wav;base64,${result.wav_b64}`}>
            Your browser does not support the audio element.
          </audio>
        ) : null}

        <div className="flex flex-wrap gap-3">
          <a
            href={downloadPathFor(result.midi_filename)}
            download={result.midi_filename}
            className="rounded-full border border-sky-400/40 bg-sky-500/10 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:border-sky-300 hover:bg-sky-500/20"
          >
            Download MIDI
          </a>
          {result.wav_filename ? (
            <a
              href={downloadPathFor(result.wav_filename)}
              download={result.wav_filename}
              className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:border-emerald-300 hover:bg-emerald-500/20"
            >
              Download WAV
            </a>
          ) : null}
        </div>
      </div>
    </Frame>
  );
}
