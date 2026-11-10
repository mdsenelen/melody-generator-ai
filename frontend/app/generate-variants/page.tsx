"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { AudioRecorder } from "../../components/audio-recorder";
import { MidiPlayer } from "../../components/midi-player";
import { Spinner } from "../../components/spinner";
import { MoodBadge } from "../../components/ui/badge";
import { buttonClass } from "../../components/ui/button";
import { Card } from "../../components/ui/card";
import { EmptyState, WarningBanner } from "../../components/ui/feedback";
import { Pills } from "../../components/ui/pills";
import { Slider, TemperatureInput } from "../../components/ui/slider";
import { Label } from "../../components/ui/text";
import { VariantPicker } from "../../components/ui/variant-picker";
import { UploadButton, type UploadSuccessPayload } from "../../components/upload-button";
import {
  runBrowserTranscription,
  type StoredUpload,
} from "../lib/browserTranscribe/runBrowserTranscription";
import { requestJson } from "../lib/request";
import { variantsResponseSchema, type VariantsResponse } from "../../src/domain/contracts/variants";
import { useSessionStore } from "../lib/session-store";
import { uploadWhenAwake } from "../lib/uploadWhenAwake";
import { usePipeline } from "../../hooks/use-pipeline";
import {
  failed,
  invalidInput,
  isCancellation,
  toGenerationError,
  type DomainError,
} from "../../src/domain/errors";
import type { AudioSource } from "../../src/domain/pipeline";
import { describeVariantsProgress } from "../../src/domain/statusCopy";

const GREEK = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ"];

function appendRetryHint(message: string, retryAfterSeconds: number | undefined): string {
  return retryAfterSeconds ? `${message} (retry in ~${retryAfterSeconds}s)` : message;
}

// A selected file or recording, once transcribed in the browser: the job
// lets generation skip server-side Basic Pitch.
type TranscribedSelection = { file: File; jobId: string; stored: StoredUpload };

// What generation can send for the selected clip.
type ClipSource =
  | { kind: "job"; jobId: string; stored: StoredUpload }
  // The browser path couldn't finish; `stored` is the upload if it happened.
  | { kind: "clip"; stored: StoredUpload | null };

function buildDefaultTemperatures(count: number) {
  if (count === 4) {
    return [0.7, 0.9, 1.0, 1.3];
  }
  return Array.from({ length: count }, (_, index) => Number((0.7 + index * 0.2).toFixed(2)));
}

export default function GenerateVariantsPage() {
  const lastUpload = useSessionStore((state) => state.lastUpload);
  const [showRecorder, setShowRecorder] = useState(false);
  // One source at a time: a fresh upload (already on the backend), a
  // recording (not uploaded yet), or the last upload kept in the session.
  const [selection, setSelection] = useState<AudioSource | null>(null);
  const [nVariants, setNVariants] = useState(4);
  const [selectedMood, setSelectedMood] = useState("Neutral");
  const [selectedScale, setSelectedScale] = useState("Original");
  const [temperatures, setTemperatures] = useState<number[]>(buildDefaultTemperatures(4));
  // An error raised before any run starts (failed upload, nothing chosen).
  const [notice, setNotice] = useState<DomainError | null>(null);
  // Kept beside the run state: a failed regeneration still shows the
  // previous variants under its error banner.
  const [result, setResult] = useState<VariantsResponse | null>(null);
  const [activeVariant, setActiveVariant] = useState(0);
  const transcribedRef = useRef<TranscribedSelection | null>(null);
  const { state: run, dispatch, start, cancel, dismissError } = usePipeline<VariantsResponse>();

  const loading = run.status === "transcribing" || run.status === "generating";
  const error = notice ?? (run.status === "failed" && !run.dismissed ? run.error : null);

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

  const clearErrors = () => {
    setNotice(null);
    dismissError();
  };

  const handleUploadSuccess = ({ id, filename, file }: UploadSuccessPayload) => {
    setSelection({ kind: "upload", file, stored: { id, filename } });
    transcribedRef.current = null;
    clearErrors();
  };

  const handleRecordingComplete = (file: File) => {
    setSelection({ kind: "recording", file });
    transcribedRef.current = null;
    clearErrors();
    setShowRecorder(false);
  };

  const handleUseStoredUpload = () => {
    if (!lastUpload) return;
    setSelection({ kind: "session", sourceName: lastUpload.sourceName });
    clearErrors();
  };

  const handleUploadError = (message: string) => setNotice(failed(message));

  // Cancel unmounts with the run, so hand focus back to Generate -- once the
  // idle state has rendered, since Generate is disabled until then.
  const generateButtonRef = useRef<HTMLButtonElement>(null);
  const focusGenerateRef = useRef(false);
  const cancelGeneration = () => {
    focusGenerateRef.current = true;
    cancel();
  };
  useEffect(() => {
    if (!loading && focusGenerateRef.current) {
      focusGenerateRef.current = false;
      generateButtonRef.current?.focus();
    }
  }, [loading]);

  // Basic Pitch inside the 60 s generation request timed out on the free
  // tier's 0.1 vCPU ("Generation timed out after 60s"). Transcribing in the
  // browser first (same path as /analyse) hands the server a job instead,
  // so it only decodes and generates. Any browser-side failure returns a
  // "clip" source and generation falls back to sending the clip as before.
  const transcribeSelection = async (
    source: AudioSource & { file: File },
    runId: number,
    signal: AbortSignal,
  ): Promise<ClipSource> => {
    const { file } = source;
    const cached = transcribedRef.current;
    if (cached?.file === file) {
      return { kind: "job", jobId: cached.jobId, stored: cached.stored };
    }
    const uploaded = source.kind === "upload" ? source.stored : null;
    let uploadOnce: Promise<StoredUpload> | null = null;
    const ensureUploaded = (): Promise<StoredUpload> => {
      if (uploaded) {
        return Promise.resolve(uploaded);
      }
      uploadOnce ??= uploadWhenAwake(file, { signal }).then(({ id, filename }) => ({
        id,
        filename,
      }));
      return uploadOnce;
    };

    try {
      const outcome = await runBrowserTranscription({
        file,
        signal,
        ensureUploaded,
        // Reported in 10% steps (see transition): the spinner label is a live region.
        onProgress: (fraction) => dispatch({ type: "browserProgress", runId, fraction }),
      });
      if (outcome.kind === "completed") {
        transcribedRef.current = { file, jobId: outcome.job.job_id, stored: outcome.stored };
        return { kind: "job", jobId: outcome.job.job_id, stored: outcome.stored };
      }
      return { kind: "clip", stored: outcome.stored };
    } catch (error) {
      if (isCancellation(error)) {
        throw error;
      }
      console.warn("[variants] browser transcription failed; sending the clip instead", {
        error,
      });
      return { kind: "clip", stored: null };
    }
  };

  const generateVariants = async () => {
    if (!selection) {
      setNotice(invalidInput("Choose or record an audio clip before generating variants."));
      return;
    }

    setNotice(null);
    const source = selection;
    const { runId, signal } = start(source);
    try {
      const formData = new FormData();
      const transcribed =
        source.kind === "session" ? null : await transcribeSelection(source, runId, signal);
      dispatch({ type: "generationStarted", runId });
      if (transcribed?.kind === "job") {
        formData.append("job_id", transcribed.jobId);
        formData.append("filename", transcribed.stored.filename);
        formData.append("upload_id", transcribed.stored.id);
      } else if (transcribed?.stored) {
        // The browser path uploaded it before falling back.
        formData.append("filename", transcribed.stored.filename);
        formData.append("upload_id", transcribed.stored.id);
      } else if (source.kind === "upload") {
        // Already on the backend (UploadButton uploads straight to it). This
        // request goes through a Vercel function, which rejects bodies over
        // ~4.5 MB with a 413, so send the reference, not the bytes again.
        formData.append("filename", source.stored.filename);
        formData.append("upload_id", source.stored.id);
      } else if (source.kind === "recording") {
        // A fresh recording: small, and not uploaded anywhere yet.
        formData.append("file", source.file);
      } else if (lastUpload?.jobId) {
        // Already transcribed (via /analyse) -- skip re-transcribing the
        // same audio server-side. The upload reference rides along: a job
        // registered from a browser transcription before the backend kept
        // its audio can only find that audio through it.
        formData.append("job_id", lastUpload.jobId);
        formData.append("filename", lastUpload.filename);
        formData.append("upload_id", lastUpload.uploadId);
      } else if (lastUpload) {
        formData.append("filename", lastUpload.filename);
        formData.append("upload_id", lastUpload.uploadId);
      }
      formData.append("n_variants", String(nVariants));
      formData.append("temperatures", JSON.stringify(temperatures.slice(0, nVariants)));

      // A body without a variants array fails the contract (ContractError).
      const data = await requestJson("/api/generate-variants", variantsResponseSchema, {
        method: "POST",
        body: formData,
        expectedContentType: "application/json",
        signal,
      });
      if (signal.aborted) {
        return;
      }
      setResult(data);
      setActiveVariant(0);
      dispatch({ type: "completed", runId, artifact: data });
    } catch (generationError) {
      if (isCancellation(generationError)) {
        dispatch({ type: "cancel", runId });
        return;
      }
      // busy / warming_up / restarting each render distinctly from a real
      // failure: see toGenerationError.
      dispatch({ type: "failed", runId, error: toGenerationError(generationError) });
    }
  };

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <Label>Variants</Label>
        <h1 className="font-display text-foreground mt-2 text-2xl font-light">
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
                onUploadError={handleUploadError}
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
                <AudioRecorder onRecordingComplete={handleRecordingComplete} showLivePitch />
              </div>
            ) : null}

            {lastUpload ? (
              <div className="border-border bg-background mt-4 rounded-[var(--radius)] border p-3">
                <p className="text-muted-foreground truncate text-xs">{lastUpload.sourceName}</p>
                <button
                  type="button"
                  onClick={handleUseStoredUpload}
                  disabled={selection?.kind === "session"}
                  className="border-primary/40 bg-primary/10 text-primary hover:border-primary/70 mt-3 w-full rounded-[var(--radius)] border px-3 py-2 text-left text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {selection?.kind === "session" ? "Using this upload" : "Use my last upload"}
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
            ref={generateButtonRef}
            onClick={generateVariants}
            aria-label="Generate variants"
            disabled={loading || !selection}
            className={buttonClass("primary", { className: "mt-2 w-full" })}
          >
            {loading ? (
              <Spinner size="sm" label={describeVariantsProgress(run)} />
            ) : (
              `Generate ${nVariants} Variants`
            )}
          </button>
          {loading ? (
            <button
              type="button"
              onClick={cancelGeneration}
              className={buttonClass("secondary", { className: "mt-2 w-full" })}
            >
              Cancel
            </button>
          ) : null}
        </Card>

        {/* Results take the placeholder's slot (right of the controls on wide
            screens, below them on narrow ones) instead of rendering below an
            empty 520px panel. */}
        {result ? (
          <Card className="self-start">
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
                  Heads up: running a full transcription right after generating variants can make
                  the server restart. If that happens, wait a few minutes. Your clip is picked up
                  again automatically. If it has not finished after about 20 minutes, upload it
                  again.
                </p>
              </div>
              {mood ? <MoodBadge mood={mood} label={`Mood: ${mood}`} /> : null}
            </div>

            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <Card variant="muted" className="text-muted-foreground text-sm">
                <p className="text-foreground">Checkpoint status</p>
                <p className="mt-2">
                  CVAE loaded: {result.model_status.cvae.loaded ? "yes" : "no"}
                </p>
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
        ) : (
          <Card className="min-h-[520px]" variant="muted">
            <EmptyState
              title="Configure on the left"
              body="Load audio, set transformation parameters, and generate."
            />
          </Card>
        )}
      </div>

      {error ? (
        <div aria-live="polite">
          {error.code === "warming_up" ? (
            <div data-testid="generation-error-warming_up">
              <WarningBanner
                title="Server is starting up"
                body={appendRetryHint(error.message, error.retryAfterSeconds)}
              />
            </div>
          ) : error.code === "busy" ? (
            <div data-testid="generation-error-busy">
              <WarningBanner
                title="Server is busy — try again"
                body={appendRetryHint(error.message, error.retryAfterSeconds)}
              />
            </div>
          ) : error.code === "restarting" ? (
            <div data-testid="generation-error-restarting">
              <WarningBanner title="Server restarted" body={error.message} />
            </div>
          ) : (
            <div
              className="rounded-[var(--radius)] border border-red-500/40 bg-[#120808] p-4 text-sm text-red-100"
              data-testid="generation-error-failed"
            >
              {error.message}
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
