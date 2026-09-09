"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { AudioRecorder } from "../../components/audio-recorder";
import { Spinner } from "../../components/spinner";
import { UploadButton, type UploadSuccessPayload } from "../../components/upload-button";
import { requestJson } from "../lib/request";
import { useSessionStore } from "../lib/session-store";

type Variant = {
  index: number;
  temperature: number;
  midi_b64: string;
  midi_filename: string;
  midi_download_path: string;
  wav_b64: string | null;
  wav_filename: string;
  wav_download_path: string;
};

type VariantsResponse = {
  n_variants: number;
  temperatures: number[];
  mood_idx: number;
  mood_label: "happy" | "sad" | "neutral";
  model_status: {
    cvae: { path: string; exists: boolean; size_mb: number; loaded: boolean };
    iddm_ppo: { path: string; exists: boolean; size_mb: number; loaded: boolean };
    device: string;
    load_error: string | null;
    fluidsynth_available: boolean;
  };
  variants: Variant[];
  job_id: string;
};

const moodMeta = {
  happy: {
    emoji: "😄",
    label: "Happy",
    classes: "border-yellow-500 bg-yellow-900/40 text-yellow-100",
  },
  sad: { emoji: "😢", label: "Sad", classes: "border-blue-500 bg-blue-900/40 text-blue-100" },
  neutral: { emoji: "😐", label: "Neutral", classes: "border-gray-600 bg-gray-800 text-gray-100" },
} as const;

function buildDefaultTemperatures(count: number) {
  if (count === 4) {
    return [0.7, 0.9, 1.0, 1.3];
  }
  return Array.from({ length: count }, (_, index) => Number((0.7 + index * 0.2).toFixed(2)));
}

export default function GenerateVariantsPage() {
  const lastUpload = useSessionStore((state) => state.lastUpload);
  const [showRecorder, setShowRecorder] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedSourceName, setSelectedSourceName] = useState<string | null>(null);
  const [storedFilename, setStoredFilename] = useState<string | null>(null);
  const [useStoredUpload, setUseStoredUpload] = useState(false);
  const [nVariants, setNVariants] = useState(4);
  const [selectedMood, setSelectedMood] = useState("Neutral");
  const [selectedScale, setSelectedScale] = useState("Original");
  const [temperatures, setTemperatures] = useState<number[]>(buildDefaultTemperatures(4));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<VariantsResponse | null>(null);
  const [activeVariant, setActiveVariant] = useState(0);

  const mood = useMemo(() => (result ? moodMeta[result.mood_label] : null), [result]);

  const updateVariantCount = (count: number) => {
    setNVariants(count);
    setTemperatures((current) => {
      const defaults = buildDefaultTemperatures(count);
      return defaults.map((fallback, index) => current[index] ?? fallback);
    });
    if (activeVariant >= count) {
      setActiveVariant(0);
    }
  };

  const updateTemperature = (index: number, value: number) => {
    setTemperatures((current) =>
      current.map((entry, entryIndex) => (entryIndex === index ? value : entry)),
    );
  };

  const handleUploadSuccess = ({ filename, file }: UploadSuccessPayload) => {
    setSelectedFile(file);
    setSelectedSourceName(file.name);
    setStoredFilename(filename);
    setUseStoredUpload(false);
    setError(null);
  };

  const handleRecordingComplete = (file: File) => {
    setSelectedFile(file);
    setSelectedSourceName(file.name);
    setStoredFilename(null);
    setUseStoredUpload(false);
    setError(null);
    setShowRecorder(false);
  };

  const handleUseStoredUpload = () => {
    if (!lastUpload) return;
    setSelectedFile(null);
    setSelectedSourceName(lastUpload.sourceName);
    setStoredFilename(lastUpload.filename);
    setUseStoredUpload(true);
    setError(null);
  };

  const generateVariants = async () => {
    if (!selectedFile && !useStoredUpload) {
      setError("Choose or record an audio clip before generating variants.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      if (selectedFile) {
        formData.append("file", selectedFile);
      } else if (lastUpload) {
        formData.append("filename", lastUpload.filename);
        formData.append("upload_id", lastUpload.uploadId);
      }
      formData.append("n_variants", String(nVariants));
      formData.append("temperatures", JSON.stringify(temperatures.slice(0, nVariants)));

      const data = await requestJson<VariantsResponse>("/api/generate-variants", {
        method: "POST",
        body: formData,
        expectedContentType: "application/json",
      });
      if (!Array.isArray(data.variants)) {
        throw new Error("Variant generation failed");
      }
      setResult(data);
      setActiveVariant(0);
    } catch (generationError) {
      setError(
        generationError instanceof Error ? generationError.message : "Variant generation failed",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 pb-8">
      <div>
        <p className="text-[11px] font-medium tracking-[0.24em] text-[#bf9bff] uppercase">
          Variants
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-[-0.03em] text-white sm:text-4xl">
          Transform your audio
        </h1>
        <p className="mt-2 max-w-xl text-sm text-white/55">
          Configure the source and transformation controls, then generate new playable ideas.
        </p>
      </div>

      <div className="grid gap-6 xl:grid-cols-[280px_minmax(0,1fr)]">
        <section className="rounded-[1.5rem] border border-white/10 bg-[rgba(17,22,32,0.78)] p-5 shadow-[0_12px_28px_rgba(2,6,23,0.2)] backdrop-blur-md">
          <div className="mb-5 text-[10px] font-semibold tracking-[0.22em] text-white/40 uppercase">
            Audio source
          </div>
          <div className="space-y-3">
            <UploadButton
              onUploadSuccess={handleUploadSuccess}
              onUploadError={setError}
              label="Upload audio"
            />
            <button
              type="button"
              onClick={() => setShowRecorder((v) => !v)}
              className="w-full rounded-xl border border-white/10 bg-white/[0.04] px-4 py-3 text-sm font-semibold text-white/75 transition hover:border-white/25 hover:text-white"
            >
              {showRecorder ? "Hide recorder" : "Record audio"}
            </button>
          </div>

          {showRecorder ? (
            <div className="mt-4">
              <AudioRecorder onRecordingComplete={handleRecordingComplete} />
            </div>
          ) : null}

          {lastUpload ? (
            <div className="mt-4 rounded-xl border border-white/10 bg-white/[0.03] p-3">
              <p className="truncate text-xs text-white/50">{lastUpload.sourceName}</p>
              <button
                type="button"
                onClick={handleUseStoredUpload}
                disabled={useStoredUpload}
                className="mt-3 w-full rounded-lg border border-[#a879ff]/40 bg-[#8b5cf6]/10 px-3 py-2 text-left text-sm font-semibold text-[#d7c1ff] transition hover:border-[#bda0ff] disabled:cursor-not-allowed disabled:opacity-60"
              >
                {useStoredUpload ? "Using this upload" : "Use my last upload"}
              </button>
            </div>
          ) : null}

          <div className="mt-8 border-t border-white/10 pt-6">
            <p className="text-[10px] font-semibold tracking-[0.22em] text-white/40 uppercase">
              Transformation
            </p>
            <div className="mt-4 space-y-4">
              <div>
                <p className="mb-2 text-xs text-white/45">Mood</p>
                <div className="flex flex-wrap gap-2">
                  {["Happy", "Neutral", "Sad"].map((moodOption) => (
                    <button
                      key={moodOption}
                      type="button"
                      onClick={() => setSelectedMood(moodOption)}
                      className={`rounded-lg border px-3 py-2 text-xs transition ${selectedMood === moodOption ? "border-[#a879ff]/70 bg-[#8b5cf6]/25 text-white" : "border-white/10 bg-white/[0.03] text-white/50 hover:border-white/25"}`}
                    >
                      {moodOption}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <p className="mb-2 text-xs text-white/45">Scale</p>
                <div className="flex flex-wrap gap-2">
                  {["Original", "Major", "Minor"].map((scaleOption) => (
                    <button
                      key={scaleOption}
                      type="button"
                      onClick={() => setSelectedScale(scaleOption)}
                      className={`rounded-lg border px-3 py-2 text-xs transition ${selectedScale === scaleOption ? "border-[#a879ff]/70 bg-[#8b5cf6]/25 text-white" : "border-white/10 bg-white/[0.03] text-white/50 hover:border-white/25"}`}
                    >
                      {scaleOption}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6 border-t border-white/10 pt-6">
            <label className="flex flex-col gap-2 text-sm text-white/55">
              <span className="flex items-center justify-between">
                <span className="text-[10px] font-semibold tracking-[0.2em] text-white/40 uppercase">
                  Number of variants
                </span>
                <span className="font-medium text-white">{nVariants}</span>
              </span>
              <input
                type="range"
                min={1}
                max={8}
                step={1}
                value={nVariants}
                onChange={(event) => updateVariantCount(Number(event.target.value))}
                className="h-1 w-full cursor-pointer accent-[#a879ff]"
              />
            </label>
          </div>

          <button
            type="button"
            onClick={generateVariants}
            aria-label="Generate variants"
            disabled={loading || (!selectedFile && !useStoredUpload)}
            className="mt-8 inline-flex w-full items-center justify-center rounded-xl border border-[#a879ff]/70 bg-[#8b5cf6]/30 px-4 py-3 text-sm font-semibold text-white transition hover:bg-[#8b5cf6]/45 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? <Spinner size="sm" label="Generating variants" /> : "Generate 3 Variants"}
          </button>
        </section>

        <section className="min-h-[520px] rounded-[1.5rem] border border-white/10 bg-[rgba(17,22,32,0.56)] p-6 shadow-[0_12px_28px_rgba(2,6,23,0.16)] backdrop-blur-md">
          {!result ? (
            <div className="flex min-h-[470px] flex-col items-center justify-center text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-full border border-white/15 bg-white/[0.04] text-3xl font-light text-white/55">
                +
              </div>
              <h2 className="mt-6 text-xl font-semibold text-white">Configure on the left</h2>
              <p className="mt-3 max-w-sm text-sm leading-6 text-white/45">
                Load audio, set transformation parameters, and generate.
              </p>
            </div>
          ) : null}
        </section>
      </div>

      {error ? (
        <div className="rounded-2xl border border-red-500/40 bg-red-950/40 p-4 text-sm text-red-100">
          {error}
        </div>
      ) : null}

      {result ? (
        <section className="rounded-[2rem] border border-white/10 bg-gray-900/80 p-6 shadow-xl shadow-black/20">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-2xl font-semibold text-white">Rendered variants</h2>
              <p className="mt-2 text-sm text-gray-400">
                Device: {result.model_status.device}. FluidSynth available:{" "}
                {result.model_status.fluidsynth_available ? "yes" : "no"}.
              </p>
            </div>
            {mood ? (
              <div
                className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-semibold ${mood.classes}`}
              >
                <span>{mood.emoji}</span>
                <span>Mood: {mood.label}</span>
              </div>
            ) : null}
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="rounded-3xl border border-white/10 bg-black/20 p-4 text-sm text-gray-300">
              <p className="font-semibold text-white">Checkpoint status</p>
              <p className="mt-2">CVAE loaded: {result.model_status.cvae.loaded ? "yes" : "no"}</p>
              <p>IDDM-PPO loaded: {result.model_status.iddm_ppo.loaded ? "yes" : "no"}</p>
              <p className="mt-2 text-xs text-gray-500">
                CVAE {result.model_status.cvae.size_mb} MB, IDDM-PPO{" "}
                {result.model_status.iddm_ppo.size_mb} MB
              </p>
            </div>
            <div className="rounded-3xl border border-white/10 bg-black/20 p-4 text-sm text-gray-300">
              <p className="font-semibold text-white">Variant controls used</p>
              <p className="mt-2">{result.n_variants} variants</p>
              <p className="mt-1">Temperatures: {result.temperatures.join(", ")}</p>
              {result.model_status.load_error ? (
                <p className="mt-2 text-red-300">{result.model_status.load_error}</p>
              ) : null}
            </div>
          </div>

          <div className="mt-6 flex flex-wrap gap-2">
            {result.variants.map((variant, index) => (
              <button
                key={variant.index}
                type="button"
                onClick={() => setActiveVariant(index)}
                className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                  activeVariant === index
                    ? "border border-purple-400/60 bg-purple-500/20 text-white"
                    : "border border-white/10 bg-white/5 text-gray-300 hover:border-white/20 hover:text-white"
                }`}
              >
                Variant {index + 1}
              </button>
            ))}
          </div>

          {result.variants[activeVariant] ? (
            <div className="mt-6 rounded-3xl border border-white/10 bg-black/20 p-5">
              <p className="text-sm text-gray-400">
                Temperature:{" "}
                <span className="font-semibold text-white">
                  {result.variants[activeVariant].temperature}
                </span>
              </p>
              {result.variants[activeVariant].wav_b64 ? (
                <audio
                  controls
                  className="mt-4 w-full"
                  src={`data:audio/wav;base64,${result.variants[activeVariant].wav_b64}`}
                >
                  Your browser does not support the audio element.
                </audio>
              ) : (
                <p className="mt-4 text-sm text-gray-400">
                  WAV preview is unavailable because FluidSynth could not render this variant in the
                  current backend environment.
                </p>
              )}
              <div className="mt-4 flex flex-wrap gap-3">
                <Link
                  href={`/result/${result.job_id}`}
                  className="rounded-full border border-sky-400/40 bg-sky-500/10 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:border-sky-300 hover:bg-sky-500/20"
                >
                  View &amp; download result
                </Link>
              </div>
            </div>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}
