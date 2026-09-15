"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { AudioRecorder } from "../../components/audio-recorder";
import { MidiPlayer } from "../../components/midi-player";
import { Spinner } from "../../components/spinner";
import { MoodBadge } from "../../components/ui/badge";
import { buttonClass } from "../../components/ui/button";
import { Card } from "../../components/ui/card";
import { EmptyState } from "../../components/ui/feedback";
import { Pills } from "../../components/ui/pills";
import { Slider, TemperatureInput } from "../../components/ui/slider";
import { Label } from "../../components/ui/text";
import { VariantPicker } from "../../components/ui/variant-picker";
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

const GREEK = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ"];

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

  const mood = useMemo(() => (result ? result.mood_label : null), [result]);

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
      } else if (lastUpload?.jobId) {
        // Already transcribed (via /analyse) -- skip re-transcribing the
        // same audio server-side.
        formData.append("job_id", lastUpload.jobId);
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
    <div className="space-y-8">
      <div>
        <Label>Variants</Label>
        <h1 className="font-display text-foreground mt-2 text-3xl font-light">
          Transform your audio
        </h1>
        <p className="text-muted-foreground mt-2 max-w-xl text-sm">
          Configure the source and transformation controls, then generate new playable ideas.
        </p>
      </div>

      <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)]">
        <Card className="space-y-6">
          <div>
            <Label>Audio source</Label>
            <div className="mt-3 space-y-3">
              <UploadButton
                onUploadSuccess={handleUploadSuccess}
                onUploadError={setError}
                label="Upload audio"
              />
              <button
                type="button"
                onClick={() => setShowRecorder((v) => !v)}
                className={buttonClass("ghost", { className: "w-full" })}
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
              <div className="border-border bg-background mt-4 rounded-[var(--radius)] border p-3">
                <p className="text-muted-foreground truncate text-xs">{lastUpload.sourceName}</p>
                <button
                  type="button"
                  onClick={handleUseStoredUpload}
                  disabled={useStoredUpload}
                  className="border-primary/40 bg-primary/10 text-primary hover:border-primary/70 mt-3 w-full rounded-[var(--radius)] border px-3 py-2 text-left text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {useStoredUpload ? "Using this upload" : "Use my last upload"}
                </button>
              </div>
            ) : null}
          </div>

          <div className="border-border space-y-4 border-t pt-6">
            <Label>Transformation</Label>
            <div>
              <p className="text-muted-foreground mb-2 text-xs">Mood</p>
              <Pills
                options={["Happy", "Neutral", "Sad"]}
                value={selectedMood}
                onChange={setSelectedMood}
                small
              />
            </div>
            <div>
              <p className="text-muted-foreground mb-2 text-xs">Scale</p>
              <Pills
                options={["Original", "Major", "Minor"]}
                value={selectedScale}
                onChange={setSelectedScale}
                small
              />
            </div>
          </div>

          <div className="border-border border-t pt-6">
            <Slider
              label="Number of variants"
              value={nVariants}
              min={1}
              max={8}
              onChange={updateVariantCount}
            />
          </div>

          <div className="border-border space-y-3 border-t pt-6">
            <div className="flex items-center justify-between">
              <Label>Temperature per variant</Label>
              <span className="text-muted-foreground font-mono text-[9px]">0.3 – 2.0</span>
            </div>
            <div className="space-y-3">
              {Array.from({ length: nVariants }, (_, index) => (
                <TemperatureInput
                  key={index}
                  label={GREEK[index]}
                  value={temperatures[index] ?? 1.0}
                  onChange={(value) => updateTemperature(index, value)}
                />
              ))}
            </div>
          </div>

          <button
            type="button"
            onClick={generateVariants}
            aria-label="Generate variants"
            disabled={loading || (!selectedFile && !useStoredUpload)}
            className={buttonClass("primary", { className: "mt-2 w-full" })}
          >
            {loading ? (
              <Spinner size="sm" label="Generating variants" />
            ) : (
              `Generate ${nVariants} Variants`
            )}
          </button>
        </Card>

        <Card className="min-h-[520px]" variant="muted">
          {!result ? (
            <EmptyState
              title="Configure on the left"
              body="Load audio, set transformation parameters, and generate."
            />
          ) : null}
        </Card>
      </div>

      {error ? (
        <div className="rounded-[var(--radius)] border border-red-500/40 bg-[#120808] p-4 text-sm text-red-100">
          {error}
        </div>
      ) : null}

      {result ? (
        <Card>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="font-display text-foreground text-2xl font-light">
                Rendered variants
              </h2>
              <p className="text-muted-foreground mt-2 text-sm">
                Generated on {result.model_status.device}. Play each melody below, or download the
                MIDI.
              </p>
              <p className="text-muted-foreground mt-1 text-xs">
                Running a full transcription right after generating may take a little longer while
                the server recycles memory.
              </p>
            </div>
            {mood ? <MoodBadge mood={mood} label={`Mood: ${mood}`} /> : null}
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <Card variant="muted" className="text-muted-foreground text-sm">
              <p className="text-foreground">Checkpoint status</p>
              <p className="mt-2">CVAE loaded: {result.model_status.cvae.loaded ? "yes" : "no"}</p>
              <p>IDDM-PPO loaded: {result.model_status.iddm_ppo.loaded ? "yes" : "no"}</p>
              <p className="text-muted-foreground mt-2 font-mono text-xs">
                CVAE {result.model_status.cvae.size_mb} MB, IDDM-PPO{" "}
                {result.model_status.iddm_ppo.size_mb} MB
              </p>
            </Card>
            <Card variant="muted" className="text-muted-foreground text-sm">
              <p className="text-foreground">Variant controls used</p>
              <p className="mt-2">{result.n_variants} variants</p>
              <p className="mt-1">Temperatures: {result.temperatures.join(", ")}</p>
              {result.model_status.load_error ? (
                <p className="mt-2 text-red-300">{result.model_status.load_error}</p>
              ) : null}
            </Card>
          </div>

          <div className="mt-6">
            <VariantPicker
              count={result.variants.length}
              active={activeVariant}
              onSelect={setActiveVariant}
            />
          </div>

          {result.variants[activeVariant] ? (
            <div className="border-border bg-background mt-6 rounded-[var(--radius)] border p-5">
              <p className="text-muted-foreground text-sm">
                Temperature:{" "}
                <span className="text-foreground">
                  {result.variants[activeVariant].temperature}
                </span>
              </p>
              <MidiPlayer
                key={result.variants[activeVariant].index}
                midiB64={result.variants[activeVariant].midi_b64}
                className="mt-4"
              />
              <div className="mt-4 flex flex-wrap gap-3">
                <Link href={`/result/${result.job_id}`} className={buttonClass("secondary")}>
                  View &amp; download result
                </Link>
              </div>
            </div>
          ) : null}
        </Card>
      ) : null}
    </div>
  );
}
