"use client";

import { useMemo, useState } from "react";

import { AudioRecorder } from "../../components/audio-recorder";
import { UploadButton, type UploadSuccessPayload } from "../../components/upload-button";
import { requestJson } from "../lib/request";

type TabKey = "upload" | "record";

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
};

const moodMeta = {
  happy: {
    emoji: "Happy",
    classes: "border-yellow-500 bg-yellow-900/40 text-yellow-100",
  },
  sad: {
    emoji: "Sad",
    classes: "border-blue-500 bg-blue-900/40 text-blue-100",
  },
  neutral: {
    emoji: "Neutral",
    classes: "border-gray-600 bg-gray-800 text-gray-100",
  },
} as const;

function buildDefaultTemperatures(count: number) {
  if (count === 4) {
    return [0.7, 0.9, 1.0, 1.3];
  }
  return Array.from({ length: count }, (_, index) => Number((0.7 + index * 0.2).toFixed(2)));
}

function triggerBase64Download(filename: string, data: string, mimeType: string) {
  const anchor = document.createElement("a");
  anchor.href = `data:${mimeType};base64,${data}`;
  anchor.download = filename;
  anchor.click();
}

export default function GenerateVariantsPage() {
  const [activeTab, setActiveTab] = useState<TabKey>("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedSourceName, setSelectedSourceName] = useState<string | null>(null);
  const [storedFilename, setStoredFilename] = useState<string | null>(null);
  const [nVariants, setNVariants] = useState(4);
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
    setTemperatures((current) => current.map((entry, entryIndex) => (entryIndex === index ? value : entry)));
  };

  const handleUploadSuccess = ({ filename, file }: UploadSuccessPayload) => {
    setSelectedFile(file);
    setSelectedSourceName(file.name);
    setStoredFilename(filename);
    setError(null);
  };

  const handleRecordingComplete = (file: File) => {
    setSelectedFile(file);
    setSelectedSourceName(file.name);
    setStoredFilename(null);
    setError(null);
  };

  const generateVariants = async () => {
    if (!selectedFile) {
      setError("Choose or record an audio clip before generating variants.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
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
      setError(generationError instanceof Error ? generationError.message : "Variant generation failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">


      <section className="rounded-[2rem] border border-white/10 bg-gray-900/80 p-6 shadow-xl shadow-black/20">
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => setActiveTab("upload")}
            className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
              activeTab === "upload"
                ? "border border-purple-400/60 bg-purple-500/20 text-white"
                : "border border-white/10 bg-white/5 text-gray-300 hover:border-white/20 hover:text-white"
            }`}
          >
            Upload file
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("record")}
            className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
              activeTab === "record"
                ? "border border-purple-400/60 bg-purple-500/20 text-white"
                : "border border-white/10 bg-white/5 text-gray-300 hover:border-white/20 hover:text-white"
            }`}
          >
            Record audio
          </button>
        </div>

        <div className="mt-6">
          {activeTab === "upload" ? (
            <UploadButton onUploadSuccess={handleUploadSuccess} onUploadError={setError} label="Upload audio" />
          ) : (
            <AudioRecorder onRecordingComplete={handleRecordingComplete} />
          )}
        </div>

        <div className="mt-6 rounded-3xl border border-white/10 bg-black/20 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-white">Selected source</p>
              <p className="mt-2 text-sm text-gray-300">{selectedSourceName ?? "No audio selected yet"}</p>
              {storedFilename ? <p className="mt-1 text-xs uppercase tracking-[0.2em] text-gray-500">{storedFilename}</p> : null}
            </div>
            <button
              type="button"
              onClick={generateVariants}
              disabled={loading || !selectedFile}
              className="rounded-2xl border border-purple-400/40 bg-purple-500/15 px-5 py-3 text-sm font-semibold text-purple-100 transition hover:border-purple-300 hover:bg-purple-500/20 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Generating..." : "Generate variants"}
            </button>
          </div>

          <div className="mt-6 grid gap-3 md:grid-cols-2 xl:grid-cols-[auto_1fr]">
            <label className="flex flex-col gap-2 text-sm text-gray-300">
              <span className="font-medium text-gray-200">Variants: {nVariants}</span>
              <input
                type="range"
                min={1}
                max={8}
                step={1}
                value={nVariants}
                onChange={(event) => updateVariantCount(Number(event.target.value))}
                className="accent-purple-400"
              />
            </label>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              {temperatures.slice(0, nVariants).map((temperature, index) => (
                <label key={`temperature-${index}`} className="flex flex-col gap-2 text-sm text-gray-300">
                  <span className="font-medium text-gray-200">Temperature {index + 1}</span>
                  <input
                    type="number"
                    min={0.3}
                    max={2.0}
                    step={0.1}
                    value={temperature}
                    onChange={(event) => updateTemperature(index, Number(event.target.value))}
                    className="rounded-2xl border border-white/10 bg-gray-950 px-3 py-2 text-white outline-none transition focus:border-purple-400"
                  />
                </label>
              ))}
            </div>
          </div>
        </div>
      </section>

      {error ? (
        <div className="rounded-2xl border border-red-500/40 bg-red-950/40 p-4 text-sm text-red-100">{error}</div>
      ) : null}

      {result ? (
        <section className="rounded-[2rem] border border-white/10 bg-gray-900/80 p-6 shadow-xl shadow-black/20">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-2xl font-semibold text-white">Rendered variants</h2>
              <p className="mt-2 text-sm text-gray-400">
                Device: {result.model_status.device}. FluidSynth available: {result.model_status.fluidsynth_available ? "yes" : "no"}.
              </p>
            </div>
            {mood ? (
              <div className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-semibold ${mood.classes}`}>
                <span>{mood.emoji}</span>
                <span>{result.mood_label}</span>
              </div>
            ) : null}
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="rounded-3xl border border-white/10 bg-black/20 p-4 text-sm text-gray-300">
              <p className="font-semibold text-white">Checkpoint status</p>
              <p className="mt-2">CVAE loaded: {result.model_status.cvae.loaded ? "yes" : "no"}</p>
              <p>IDDM-PPO loaded: {result.model_status.iddm_ppo.loaded ? "yes" : "no"}</p>
              <p className="mt-2 text-xs text-gray-500">
                CVAE {result.model_status.cvae.size_mb} MB, IDDM-PPO {result.model_status.iddm_ppo.size_mb} MB
              </p>
            </div>
            <div className="rounded-3xl border border-white/10 bg-black/20 p-4 text-sm text-gray-300">
              <p className="font-semibold text-white">Variant controls used</p>
              <p className="mt-2">{result.n_variants} variants</p>
              <p className="mt-1">Temperatures: {result.temperatures.join(", ")}</p>
              {result.model_status.load_error ? <p className="mt-2 text-red-300">{result.model_status.load_error}</p> : null}
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
                Temperature: <span className="font-semibold text-white">{result.variants[activeVariant].temperature}</span>
              </p>
              {result.variants[activeVariant].wav_b64 ? (
                <audio controls className="mt-4 w-full" src={`data:audio/wav;base64,${result.variants[activeVariant].wav_b64}`}>
                  Your browser does not support the audio element.
                </audio>
              ) : (
                <p className="mt-4 text-sm text-gray-400">
                  WAV preview is unavailable because FluidSynth could not render this variant in the current backend environment.
                </p>
              )}
              <div className="mt-4 flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={() =>
                    triggerBase64Download(
                      result.variants[activeVariant].midi_filename,
                      result.variants[activeVariant].midi_b64,
                      "audio/midi",
                    )
                  }
                  className="rounded-full border border-sky-400/40 bg-sky-500/10 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:border-sky-300 hover:bg-sky-500/20"
                >
                  Download MIDI
                </button>
                {result.variants[activeVariant].wav_b64 ? (
                  <button
                    type="button"
                    onClick={() =>
                      triggerBase64Download(
                        result.variants[activeVariant].wav_filename,
                        result.variants[activeVariant].wav_b64!,
                        "audio/wav",
                      )
                    }
                    className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:border-emerald-300 hover:bg-emerald-500/20"
                  >
                    Download WAV
                  </button>
                ) : null}
              </div>
            </div>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}
