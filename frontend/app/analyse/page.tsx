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
  TranscribeJobLostError,
  TranscribeJobSupersededError,
  type TranscriptionResult,
} from "../lib/transcribeJob";
import { topPitchClasses } from "../../utils/pitch";
import { uploadWhenAwake, WAKING_STATUS_MESSAGE, type UploadPhase } from "../lib/uploadWhenAwake";

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
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [transcription, setTranscription] = useState<Transcription | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisState>({ status: "idle" });
  const [clipWindow, setClipWindow] = useState<ClipWindow>({ start: 0, end: DEFAULT_CLIP_SEC });
  const [previewAudioUrl, setPreviewAudioUrl] = useState<string | null>(null);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Choose a file or record a clip to start.");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const transcribeReqRef = useRef(0);
  const analysisReqRef = useRef(0);
  const analysisAbortRef = useRef<AbortController | null>(null);
  const transcribeAbortRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);

  const dismissError = useCallback(() => setErrorMessage(null), []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      analysisAbortRef.current?.abort();
      transcribeAbortRef.current?.abort();
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

  const runTranscription = async (
    file: File,
    uploaded: { id: string; filename: string } | null,
    pickedAt: number,
  ) => {
    const requestId = ++transcribeReqRef.current;
    analysisReqRef.current += 1; // invalidate any in-flight analysis for the previous clip
    analysisAbortRef.current?.abort(); // and stop it actually retrying in the background
    setErrorMessage(null);
    setTranscription(null);
    setAnalysis({ status: "idle" });
    setIsTranscribing(true);
    setStatusMessage(`Transcribing ${file.name}...`);

    transcribeAbortRef.current?.abort();
    const controller = new AbortController();
    transcribeAbortRef.current = controller;
    const isCancelled = () => requestId !== transcribeReqRef.current || !mountedRef.current;

    // Set only while the recording upload below is in flight. The slow-timer
    // copy is about transcription, so it mustn't replace the waking status.
    let uploadPhase: UploadPhase | null = null;
    // While Basic Pitch runs in the browser, the status shows its progress;
    // the upload runs alongside it silently, and the slow-timer's
    // cold-server copy doesn't apply.
    let inBrowser = false;

    const slowTimer = window.setTimeout(() => {
      if (requestId === transcribeReqRef.current && uploadPhase !== "waking" && !inBrowser) {
        setStatusMessage(
          `Still transcribing ${file.name}... this can take a minute or two if the server was idle.`,
        );
      }
    }, 8000);

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
          signal: controller.signal,
          onPhase: (phase) => {
            uploadPhase = phase;
            if (!isCancelled() && !inBrowser) {
              setStatusMessage(
                phase === "waking" ? WAKING_STATUS_MESSAGE : `Uploading ${file.name}...`,
              );
            }
          },
        }).then(({ id, filename }) => {
          uploadPhase = null;
          return { id, filename };
        });
        return uploadOnce;
      };

      let lastProgressStep = -1;
      inBrowser = true;
      const browser = await runBrowserTranscription({
        file,
        signal: controller.signal,
        ensureUploaded,
        // 10% steps: the status is a live region, and every Basic Pitch
        // batch would otherwise be a screen-reader announcement.
        onProgress: (fraction) => {
          const step = Math.min(10, Math.floor(fraction * 10));
          if (step === lastProgressStep || isCancelled()) {
            return;
          }
          lastProgressStep = step;
          setStatusMessage(
            step === 10
              ? `Transcribed ${file.name} in your browser. Saving the notes...`
              : `Transcribing ${file.name} in your browser... ${step * 10}%`,
          );
        },
        onRegisterRetrying: () => {
          if (!isCancelled()) {
            setStatusMessage(`Server was still waking up — saving the notes for ${file.name}...`);
          }
        },
      });
      inBrowser = false;

      let stored: StoredUpload;
      let jobId: string;
      let result: TranscriptionResult;
      const path: TranscribePath = browser.kind === "completed" ? "browser" : "server";
      if (browser.kind === "completed") {
        window.clearTimeout(slowTimer);
        ({ stored, result } = browser);
        jobId = browser.job.job_id;
      } else {
        stored = browser.stored ?? (await ensureUploaded());
        const serverUpload = stored;
        const created = await retryOnWakeup(
          (signal) => createTranscribeJob(file, serverUpload, { signal }),
          {
            signal: controller.signal,
            onRetrying: () => {
              if (!isCancelled()) {
                setStatusMessage(`Server was still waking up — retrying ${file.name}...`);
              }
            },
          },
        );
        window.clearTimeout(slowTimer);
        jobId = created.job_id;

        if (requestId !== transcribeReqRef.current) {
          return;
        }
        setStatusMessage(`Queued for transcription...`);

        result = await pollTranscribeJob(jobId, {
          isSuperseded: () => requestId !== transcribeReqRef.current || !mountedRef.current,
          onStatusChange: (status, elapsedMs) => {
            if (requestId !== transcribeReqRef.current || !mountedRef.current) {
              return;
            }
            // The worker bounces a job queued for up to WARMUP_MAX_SECONDS
            // without ever marking it failed (see backend/app/jobs/worker.py)
            // -- a calm, specific status, not the generic "still queued"
            // wording, and never the red ErrorToast (that's reserved for a
            // real terminal failure, in the catch block below).
            if (status.status === "queued" && status.waiting_reason === "warming_up") {
              setStatusMessage(
                "The server is starting up. Your clip will be transcribed automatically, " +
                  "this can take about 2 to 4 minutes.",
              );
              return;
            }
            const stillWaking = elapsedMs > 8000;
            if (status.status === "queued") {
              setStatusMessage(
                stillWaking
                  ? `Still queued for ${file.name}... this can take a minute or two if the server was idle.`
                  : `Queued for transcription...`,
              );
            } else if (status.status === "processing") {
              setStatusMessage(
                stillWaking
                  ? `Still transcribing ${file.name}... longer clips take longer, and a cold server can add a minute or two on top.`
                  : `Transcribing ${file.name}...`,
              );
            }
          },
          onRestarting: () => {
            if (requestId !== transcribeReqRef.current || !mountedRef.current) {
              return;
            }
            // Not the red ErrorToast -- the job survives the restart in
            // Postgres, so this is still a normal in-progress state, just a
            // calmer explanation than a generic "still transcribing" while
            // we keep polling underneath.
            setStatusMessage(
              "The server restarted and is coming back up. Your clip is safe -- we'll keep checking automatically.",
            );
          },
        });
      }

      if (requestId !== transcribeReqRef.current || !mountedRef.current) {
        return;
      }

      setTranscription({
        ...result,
        sourceName: file.name,
        uploadedFilename: stored.filename,
        jobId,
      });
      setStatusMessage(`Transcription ready for ${file.name}.`);
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
      if (
        transcribeError instanceof TranscribeJobSupersededError ||
        transcribeError instanceof RetryCancelledError ||
        // Cancelling the browser path (new clip, or leaving the page)
        // terminates the worker and rejects with a plain AbortError.
        (transcribeError instanceof DOMException && transcribeError.name === "AbortError")
      ) {
        return;
      }
      console.error("[analysis] failed", { file: file.name, error: transcribeError });
      if (requestId === transcribeReqRef.current && mountedRef.current) {
        if (transcribeError instanceof TranscribeJobLostError) {
          // Calm status text, not the red ErrorToast -- a restart losing
          // the job is a known, explainable server event, not a failure
          // on the user's end.
          setStatusMessage(transcribeError.message);
        } else {
          setErrorMessage(
            transcribeError instanceof Error ? transcribeError.message : "Transcription failed",
          );
          setStatusMessage("We couldn't transcribe that audio clip.");
        }
      }
    } finally {
      if (requestId === transcribeReqRef.current && mountedRef.current) {
        setIsTranscribing(false);
      }
    }
  };

  const handleUploadSuccess = ({ id, filename, file, pickedAt }: UploadSuccessPayload) => {
    setSelectedFile(file);
    setErrorMessage(null);
    setStatusMessage(`Uploaded ${file.name}. Starting transcription...`);
    void runTranscription(file, { id, filename }, pickedAt);
  };

  const handleRecordingComplete = (file: File) => {
    // A recording has no pick; timing starts when it's handed over.
    const pickedAt = performance.now();
    setShowRecorder(true);
    setSelectedFile(file);
    setErrorMessage(null);
    setStatusMessage(`Recording ready. Transcribing ${file.name}...`);
    void runTranscription(file, null, pickedAt);
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
      <>
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

        <main className="space-y-8">
          <section className="grid gap-6 xl:grid-cols-2">
            <Card>
              <Label>Upload audio</Label>
              <div className="mt-4">
                <UploadButton
                  onUploadSuccess={handleUploadSuccess}
                  onUploadError={setErrorMessage}
                  label="Upload Audio"
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
            {isTranscribing ? (
              <Card>
                <div className="flex items-center justify-center py-14">
                  <Spinner size="lg" label={statusMessage} />
                </div>
              </Card>
            ) : null}

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
        </main>

        {errorMessage ? <ErrorToast message={errorMessage} onDismiss={dismissError} /> : null}
      </>
    </ErrorBoundary>
  );
}
