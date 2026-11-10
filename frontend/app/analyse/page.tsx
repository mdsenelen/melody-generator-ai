"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { AudioRecorder } from "../../components/audio-recorder";
import { ChordDiagram } from "../../components/chord-diagram";
import { ClipRange, type ClipWindow } from "../../components/clip-range";
import ErrorBoundary from "../../components/error-boundary";
import { ErrorToast } from "../../components/error-toast";
import { Spinner } from "../../components/spinner";
import { buttonClass } from "../../components/ui/button";
import { Card } from "../../components/ui/card";
import { WarningBanner } from "../../components/ui/feedback";
import { ChordSequence } from "../../components/ui/chord-sequence";
import { MoodBadge, StatusBadge } from "../../components/ui/badge";
import { PitchHistogram } from "../../components/ui/pitch-histogram";
import { Label, SectionHeading } from "../../components/ui/text";
import { UploadButton, type UploadSuccessPayload } from "../../components/upload-button";
import { analyzeClip, type ClipAnalysis } from "../lib/analyzeClip";
import {
  runBrowserTranscription,
  type StoredUpload,
} from "../lib/browserTranscribe/runBrowserTranscription";
import { formatTranscribeLog, type TranscribePath } from "../lib/transcribeTiming";
import { retryOnWakeup, RetryCancelledError } from "../lib/retryOnWakeup";
import { useSessionStore } from "../lib/session-store";
import {
  createTranscribeJob,
  pollTranscribeJob,
  type TranscriptionResult,
} from "../lib/transcribeJob";
import { topPitchClasses } from "../../utils/pitch";
import { uploadWhenAwake } from "../lib/uploadWhenAwake";
import { usePipeline } from "../../hooks/use-pipeline";
import { isCancellation, toTranscriptionError } from "../../src/domain/errors";
import { SLOW_AFTER_MS, sourceFile, type AudioSource } from "../../src/domain/pipeline";
import { describeTranscription } from "../../src/domain/statusCopy";

type Transcription = TranscriptionResult & {
  sourceName: string;
  uploadedFilename: string;
  jobId: string;
};

type AnalysisState =
  | { status: "idle" }
  | { status: "loading"; message?: string }
  | { status: "ready"; data: ClipAnalysis }
  | { status: "error"; error: string };

const DEFAULT_CLIP_SEC = 60;

function formatDuration(seconds: number) {
  const totalSeconds = Math.round(seconds);
  const minutes = Math.floor(totalSeconds / 60);
  const remainingSeconds = totalSeconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
}

function summariseChords(chords: string[]) {
  // detected_chords is already in playback order (one label per analysis
  // window). Keep that order for the progression view; collapse immediate
  // repeats so "C C C Am" reads as "C → Am".
  const sequence: string[] = [];
  for (const chord of chords) {
    if (sequence[sequence.length - 1] !== chord) {
      sequence.push(chord);
    }
  }

  const counts = new Map<string, number>();
  for (const chord of chords) {
    counts.set(chord, (counts.get(chord) ?? 0) + 1);
  }
  const mostCommon = [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([chord, count]) => ({ chord, count }));

  return { sequence, mostCommon };
}

function createAudioObjectUrl(base64Audio: string, mimeType: string) {
  const bytes = Uint8Array.from(atob(base64Audio), (character) => character.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type: mimeType }));
}

export default function AnalysePage() {
  const [showRecorder, setShowRecorder] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisState>({ status: "idle" });
  const [clipWindow, setClipWindow] = useState<ClipWindow>({ start: 0, end: DEFAULT_CLIP_SEC });
  const [previewAudioUrl, setPreviewAudioUrl] = useState<string | null>(null);
  // An upload that failed before any transcription started (UploadButton).
  const [uploadError, setUploadError] = useState<string | null>(null);
  const {
    state: pipeline,
    dispatch,
    start,
    dismissError: dismissFailure,
  } = usePipeline<Transcription>();

  const analysisReqRef = useRef(0);
  const analysisAbortRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);

  const isTranscribing = pipeline.status === "transcribing";
  const transcription = pipeline.status === "completed" ? pipeline.artifact : null;
  const selectedFile = pipeline.status === "idle" ? null : sourceFile(pipeline.source);
  // A job lost to a restart is a known server event, not a failure on the
  // user's end, so it never raises the red toast.
  const errorMessage =
    uploadError ??
    (pipeline.status === "failed" && !pipeline.dismissed && pipeline.error.code !== "job_lost"
      ? pipeline.error.message
      : null);

  const dismissError = useCallback(() => {
    setUploadError(null);
    dismissFailure();
  }, [dismissFailure]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      analysisAbortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!transcription?.wav_b64) {
      setPreviewAudioUrl(null);
      return;
    }
    const url = createAudioObjectUrl(transcription.wav_b64, "audio/wav");
    setPreviewAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [transcription?.wav_b64]);

  const runAnalysis = useCallback(async (jobId: string, window: ClipWindow) => {
    const reqId = ++analysisReqRef.current;
    analysisAbortRef.current?.abort();
    const controller = new AbortController();
    analysisAbortRef.current = controller;
    setAnalysis({ status: "loading" });

    try {
      const data = await retryOnWakeup(
        (signal) =>
          analyzeClip(
            { job_id: jobId, clip_start_sec: window.start, clip_end_sec: window.end },
            { signal },
          ),
        {
          signal: controller.signal,
          onRetrying: () => {
            if (reqId === analysisReqRef.current && mountedRef.current) {
              setAnalysis({
                status: "loading",
                message: "Server was still waking up — retrying analysis...",
              });
            }
          },
        },
      );
      if (reqId !== analysisReqRef.current || !mountedRef.current) {
        return;
      }
      setAnalysis({ status: "ready", data });

      // Keep the session summary other pages (generate-variants) read in sync
      // with the window that's currently shown -- but never clobber the real
      // upload id / filename runTranscription stored.
      const session = useSessionStore.getState().lastUpload;
      if (session) {
        useSessionStore.getState().setLastUpload({
          ...session,
          transcription: {
            chords: data.detected_chords,
            key: data.key,
            moodLabel: data.mood_label,
            pitchHistogram: data.pitch_histogram,
          },
        });
      }
    } catch (analysisError) {
      if (
        analysisError instanceof RetryCancelledError ||
        controller.signal.aborted ||
        reqId !== analysisReqRef.current ||
        !mountedRef.current
      ) {
        return;
      }
      setAnalysis({
        status: "error",
        error: analysisError instanceof Error ? analysisError.message : "Analysis failed",
      });
    }
  }, []);

  const runTranscription = async (source: AudioSource & { file: File }, pickedAt: number) => {
    const { file } = source;
    const uploaded = source.kind === "upload" ? source.stored : null;
    analysisReqRef.current += 1; // invalidate any in-flight analysis for the previous clip
    analysisAbortRef.current?.abort(); // and stop it actually retrying in the background
    setUploadError(null);
    setAnalysis({ status: "idle" });
    // Aborts the previous run; its late events carry a stale runId and are dropped.
    const { runId, signal } = start(source);

    // The reducer ignores this on the browser path and while the server is
    // being woken: the copy is about a slow server transcription.
    const slowTimer = window.setTimeout(
      () => dispatch({ type: "stillWorking", runId }),
      SLOW_AFTER_MS,
    );

    try {
      // Recordings aren't uploaded yet (file uploads are, by UploadButton).
      // Memoised so the browser path's parallel upload and the server
      // fallback share one upload rather than sending the file twice.
      let uploadOnce: Promise<StoredUpload> | null = null;
      const ensureUploaded = (): Promise<StoredUpload> => {
        if (uploaded) {
          return Promise.resolve(uploaded);
        }
        uploadOnce ??= uploadWhenAwake(file, {
          signal,
          onPhase: (phase) => dispatch({ type: "uploadPhase", runId, phase }),
        }).then(({ id, filename }) => {
          dispatch({ type: "uploaded", runId });
          return { id, filename };
        });
        return uploadOnce;
      };

      const browser = await runBrowserTranscription({
        file,
        signal,
        ensureUploaded,
        // Reported in 10% steps (see transition): the status is a live
        // region, and every Basic Pitch batch would otherwise be a
        // screen-reader announcement.
        onProgress: (fraction) => dispatch({ type: "browserProgress", runId, fraction }),
        onRegisterRetrying: () => dispatch({ type: "savingRetry", runId }),
      });

      let stored: StoredUpload;
      let jobId: string;
      let result: TranscriptionResult;
      const path: TranscribePath = browser.kind === "completed" ? "browser" : "server";
      if (browser.kind === "completed") {
        window.clearTimeout(slowTimer);
        ({ stored, result } = browser);
        jobId = browser.job.job_id;
      } else {
        dispatch({ type: "fellBack", runId });
        stored = browser.stored ?? (await ensureUploaded());
        const serverUpload = stored;
        const created = await retryOnWakeup(
          (attemptSignal) => createTranscribeJob(file, serverUpload, { signal: attemptSignal }),
          { signal, onRetrying: () => dispatch({ type: "submitRetry", runId }) },
        );
        window.clearTimeout(slowTimer);
        jobId = created.job_id;

        if (signal.aborted) {
          return;
        }
        dispatch({ type: "submitted", runId });

        result = await pollTranscribeJob(jobId, {
          isSuperseded: () => signal.aborted,
          onStatusChange: (status, elapsedMs) =>
            dispatch({
              type: "serverStatus",
              runId,
              status: status.status,
              waitingReason: status.waiting_reason,
              elapsedMs,
            }),
          // Not the red ErrorToast -- the job survives the restart in
          // Postgres, so this is still a normal in-progress state.
          onRestarting: () => dispatch({ type: "restarting", runId }),
        });
      }

      if (signal.aborted) {
        return;
      }

      dispatch({
        type: "completed",
        runId,
        artifact: { ...result, sourceName: file.name, uploadedFilename: stored.filename, jobId },
      });
      console.info(
        formatTranscribeLog({
          path,
          backend: path === "browser" ? (result.tfjs_backend ?? null) : null,
          notes: result.n_notes,
          elapsedMs: performance.now() - pickedAt,
        }),
      );

      // generate-variants / choose-progression reference the upload by id.
      // jobId lets generate-variants skip re-transcribing this same audio.
      // The transcription summary is filled by runAnalysis right after (from
      // /api/analyze) -- the transcribe result no longer carries it.
      useSessionStore.getState().setLastUpload({
        uploadId: stored.id,
        filename: stored.filename,
        sourceName: file.name,
        jobId,
        transcription: { chords: [], key: "", moodLabel: "neutral", pitchHistogram: [] },
      });

      const defaultEnd = Math.min(DEFAULT_CLIP_SEC, Math.max(1, result.source_duration_sec));
      const window0: ClipWindow = { start: 0, end: defaultEnd };
      setClipWindow(window0);
      void runAnalysis(jobId, window0);
    } catch (transcribeError) {
      window.clearTimeout(slowTimer);
      if (isCancellation(transcribeError)) {
        dispatch({ type: "cancel", runId });
        return;
      }
      console.error("[analysis] failed", { file: file.name, error: transcribeError });
      dispatch({ type: "failed", runId, error: toTranscriptionError(transcribeError) });
    }
  };

  const handleUploadSuccess = ({ id, filename, file, pickedAt }: UploadSuccessPayload) => {
    void runTranscription({ kind: "upload", file, stored: { id, filename } }, pickedAt);
  };

  const handleRecordingComplete = (file: File) => {
    // A recording has no pick; timing starts when it's handed over.
    const pickedAt = performance.now();
    setShowRecorder(true);
    void runTranscription({ kind: "recording", file }, pickedAt);
  };

  const handleClipCommit = (next: ClipWindow) => {
    if (!transcription) return;
    setClipWindow(next);
    void runAnalysis(transcription.jobId, next);
  };

  const analysisData = analysis.status === "ready" ? analysis.data : null;
  const chordSummary = analysisData ? summariseChords(analysisData.detected_chords) : null;
  const detectedNotes = analysisData ? topPitchClasses(analysisData.pitch_histogram) : [];

  return (
    <ErrorBoundary>
      <div className="mx-auto max-w-4xl">
        <div className="mb-8 flex items-start justify-between">
          <SectionHeading sub="Upload or record a clip to extract key, tempo, pitch content, and chord structure.">
            Analyse Audio
          </SectionHeading>
          <div className="flex items-center gap-3">
            {showRecorder && !transcription ? (
              <StatusBadge variant="error" dot>
                Recording
              </StatusBadge>
            ) : null}
            {isTranscribing ? (
              <StatusBadge variant="accent" dot>
                Analysing
              </StatusBadge>
            ) : null}
            {transcription && !isTranscribing ? (
              <StatusBadge variant="success" dot>
                Complete
              </StatusBadge>
            ) : null}
          </div>
        </div>

        <div className="space-y-8">
          <section className="grid gap-6 xl:grid-cols-2">
            <Card>
              <Label>Upload audio</Label>
              <div className="mt-4">
                <UploadButton
                  onUploadSuccess={handleUploadSuccess}
                  onUploadError={setUploadError}
                />
              </div>

              {selectedFile ? (
                <div className="border-border bg-background text-secondary-foreground mt-5 rounded-[var(--radius)] border p-4 text-sm">
                  <Label>Source</Label>
                  <p className="text-foreground mt-2">{selectedFile.name}</p>
                </div>
              ) : null}
            </Card>

            <Card>
              <Label>Record audio</Label>
              <button
                type="button"
                onClick={() => setShowRecorder((current) => !current)}
                className="border-primary/45 bg-primary/6 text-foreground hover:border-primary/70 hover:bg-primary/12 mt-4 flex min-h-[220px] w-full flex-col items-center justify-center gap-4 rounded-[var(--radius)] border-2 border-dashed px-6 py-8 text-center transition-colors"
                aria-expanded={showRecorder}
              >
                <span className="border-primary/50 bg-primary/15 text-primary flex h-14 w-14 items-center justify-center rounded-full border text-2xl">
                  ●
                </span>
                <span className="font-display text-base">
                  {showRecorder ? "Hide recorder" : "Record audio"}
                </span>
                <span className="text-muted-foreground text-sm">
                  {showRecorder
                    ? "Close recording controls"
                    : "Use your microphone to record a clip"}
                </span>
              </button>

              {showRecorder ? (
                <div className="mt-5">
                  <AudioRecorder onRecordingComplete={handleRecordingComplete} showLivePitch />
                </div>
              ) : null}
            </Card>
          </section>

          <div className="space-y-6">
            {pipeline.status === "transcribing" ? (
              <Card>
                <div className="flex items-center justify-center py-14">
                  <Spinner size="lg" label={describeTranscription(pipeline)} />
                </div>
              </Card>
            ) : null}

            {/* A job lost to a restart is a known server event, not the
                user's failure: a calm banner, never the red toast. */}
            <div aria-live="polite">
              {pipeline.status === "failed" && pipeline.error.code === "job_lost" ? (
                <WarningBanner title="Server restarted" body={pipeline.error.message} />
              ) : null}
            </div>

            {/* Transcription result -- the full-length MIDI + downloads. */}
            {transcription ? (
              <Card variant="accent" className="space-y-5">
                <div>
                  <Label>Transcription</Label>
                  <h2 className="font-display text-foreground mt-1 text-2xl font-light">
                    {transcription.sourceName}
                  </h2>
                  <p className="text-muted-foreground mt-2 text-sm">
                    {transcription.n_notes} notes ·{" "}
                    {formatDuration(transcription.source_duration_sec)} of audio
                  </p>
                </div>

                {previewAudioUrl ? (
                  <audio controls className="w-full" src={previewAudioUrl}>
                    Your browser does not support the audio element.
                  </audio>
                ) : null}

                <div className="flex flex-wrap gap-3">
                  <Link
                    href={`/result/${transcription.jobId}`}
                    className={buttonClass("secondary")}
                  >
                    View &amp; download result
                  </Link>
                </div>
              </Card>
            ) : null}

            {/* Clip analysis -- re-runnable against any window of the source. */}
            {transcription ? (
              <Card className="space-y-5">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                  <div>
                    <Label>Clip analysis</Label>
                    <p className="text-muted-foreground mt-1 text-sm">
                      Mood, key, tempo and chords for a section of the audio. Pick a window and
                      re-analyse.
                    </p>
                  </div>
                  {analysisData ? (
                    <MoodBadge
                      mood={analysisData.mood_label}
                      label={`Mood: ${analysisData.mood_label}`}
                    />
                  ) : null}
                </div>

                <div className="border-border bg-background rounded-[var(--radius)] border p-5">
                  <ClipRange
                    sourceDurationSec={transcription.source_duration_sec}
                    value={clipWindow}
                    onCommit={handleClipCommit}
                    busy={analysis.status === "loading"}
                  />
                </div>

                <div aria-live="polite">
                  {analysis.status === "loading" ? (
                    <div className="flex items-center justify-center py-10">
                      <Spinner label={analysis.message ?? "Analysing this section..."} />
                    </div>
                  ) : analysis.status === "error" ? (
                    <div className="rounded-[var(--radius)] border border-red-500/40 bg-[#120808] p-4 text-sm text-red-100">
                      {analysis.error}
                    </div>
                  ) : analysisData ? (
                    <div className="space-y-6">
                      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                        <Card variant="muted">
                          <Label>Key</Label>
                          <p className="font-display text-foreground mt-3 text-lg font-light">
                            {analysisData.key}
                          </p>
                        </Card>
                        <Card variant="muted">
                          <Label>Notes in window</Label>
                          <p className="font-display text-foreground mt-3 text-lg font-light">
                            {analysisData.n_notes}
                          </p>
                        </Card>
                        <Card variant="muted">
                          <Label>Tempo</Label>
                          <p className="font-display text-foreground mt-3 text-lg font-light">
                            {analysisData.tempo_bpm} BPM
                          </p>
                        </Card>
                        <Card variant="muted">
                          <Label>Average pitch</Label>
                          <p className="font-display text-foreground mt-3 text-lg font-light">
                            {analysisData.average_pitch}
                          </p>
                        </Card>
                      </div>

                      {detectedNotes.length > 0 ? (
                        <Card variant="muted">
                          <Label>Detected notes</Label>
                          <div className="mt-3">
                            <ChordSequence chords={detectedNotes} />
                          </div>
                        </Card>
                      ) : null}

                      <Card variant="muted">
                        <Label>Pitch histogram</Label>
                        <p className="text-muted-foreground mt-1 text-sm">
                          Pitch-class balance across {formatDuration(analysisData.clip_start_sec)}–
                          {formatDuration(analysisData.clip_end_sec)}.
                        </p>
                        <div className="mt-5">
                          <PitchHistogram values={analysisData.pitch_histogram} />
                        </div>
                      </Card>

                      <Card variant="muted">
                        <Label>Chord progression</Label>
                        <p className="text-muted-foreground mt-2 text-sm">
                          The chords in playback order. Hover one to preview a guitar fingering.
                        </p>

                        {chordSummary && chordSummary.sequence.length > 0 ? (
                          <>
                            <div className="mt-4 flex flex-wrap items-center gap-2">
                              {chordSummary.sequence.map((chord, index) => (
                                <div key={`${chord}-${index}`} className="flex items-center gap-2">
                                  <ChordDiagram chord={chord} />
                                  {index < chordSummary.sequence.length - 1 ? (
                                    <span aria-hidden="true" className="text-muted-foreground">
                                      →
                                    </span>
                                  ) : null}
                                </div>
                              ))}
                            </div>

                            <p className="text-muted-foreground mt-4 text-sm">
                              Most used:{" "}
                              {chordSummary.mostCommon.slice(0, 3).map((entry, index) => (
                                <span key={entry.chord}>
                                  {index > 0 ? ", " : ""}
                                  <span className="text-foreground">{entry.chord}</span>
                                  <span className="text-muted-foreground"> ×{entry.count}</span>
                                </span>
                              ))}
                            </p>
                          </>
                        ) : (
                          <p className="text-muted-foreground mt-4 text-sm">
                            No chord labels were detected for this window.
                          </p>
                        )}
                      </Card>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-4 py-10 text-center">
                      <p className="text-muted-foreground text-sm">
                        Analysis for this window hasn&apos;t run yet.
                      </p>
                      <button
                        type="button"
                        onClick={() =>
                          transcription && void runAnalysis(transcription.jobId, clipWindow)
                        }
                        className={buttonClass("secondary", { small: true })}
                      >
                        Analyse this clip
                      </button>
                    </div>
                  )}
                </div>
              </Card>
            ) : null}
          </div>
        </div>

        {errorMessage ? <ErrorToast message={errorMessage} onDismiss={dismissError} /> : null}
      </div>
    </ErrorBoundary>
  );
}
