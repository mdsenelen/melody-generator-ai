"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { AudioRecorder } from "../../components/audio-recorder";
import { ChordDiagram } from "../../components/chord-diagram";
import { ClipRange, type ClipWindow } from "../../components/clip-range";
import ErrorBoundary from "../../components/error-boundary";
import { ErrorToast } from "../../components/error-toast";
import { Spinner } from "../../components/spinner";
import { UploadButton, type UploadSuccessPayload } from "../../components/upload-button";
import { analyzeClip, type ClipAnalysis } from "../lib/analyzeClip";
import { useSessionStore } from "../lib/session-store";
import {
  createTranscribeJob,
  pollTranscribeJob,
  TranscribeJobSupersededError,
  type TranscriptionResult,
} from "../lib/transcribeJob";
import { uploadFile } from "../lib/upload";

type Transcription = TranscriptionResult & {
  sourceName: string;
  uploadedFilename: string;
  jobId: string;
};

type AnalysisState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; data: ClipAnalysis }
  | { status: "error"; error: string };

const PITCH_CLASS_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const DEFAULT_CLIP_SEC = 60;

const moodMeta = {
  happy: {
    emoji: "😄",
    label: "happy",
    classes: "border-yellow-500 bg-yellow-900/40 text-yellow-100",
  },
  sad: { emoji: "😢", label: "sad", classes: "border-blue-500 bg-blue-900/40 text-blue-100" },
  neutral: { emoji: "😐", label: "neutral", classes: "border-gray-600 bg-gray-800 text-gray-100" },
} as const;

function formatDuration(seconds: number) {
  const totalSeconds = Math.round(seconds);
  const minutes = Math.floor(totalSeconds / 60);
  const remainingSeconds = totalSeconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
}

function groupChordsByRoot(chords: string[]) {
  return chords.reduce<Record<string, string[]>>((groups, chord) => {
    const root = chord.match(/^[A-G](?:#|b)?/)?.[0] ?? "Other";
    if (!groups[root]) {
      groups[root] = [];
    }
    groups[root].push(chord);
    return groups;
  }, {});
}

function createAudioObjectUrl(base64Audio: string, mimeType: string) {
  const bytes = Uint8Array.from(atob(base64Audio), (character) => character.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type: mimeType }));
}

function TranscribingAnimation({ statusMessage }: { statusMessage: string }) {
  return (
    <div className="flex items-center justify-center py-14">
      <Spinner size="lg" label={statusMessage} />
    </div>
  );
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
  const mountedRef = useRef(true);

  const dismissError = useCallback(() => setErrorMessage(null), []);

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
      const data = await analyzeClip(
        { job_id: jobId, clip_start_sec: window.start, clip_end_sec: window.end },
        { signal: controller.signal },
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
      if (controller.signal.aborted || reqId !== analysisReqRef.current || !mountedRef.current) {
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
  ) => {
    const requestId = ++transcribeReqRef.current;
    analysisReqRef.current += 1; // invalidate any in-flight analysis for the previous clip
    setErrorMessage(null);
    setTranscription(null);
    setAnalysis({ status: "idle" });
    setIsTranscribing(true);
    setStatusMessage(`Transcribing ${file.name}...`);

    const slowTimer = window.setTimeout(() => {
      if (requestId === transcribeReqRef.current) {
        setStatusMessage(
          `Still transcribing ${file.name}... this can take a minute or two if the server was idle.`,
        );
      }
    }, 8000);

    try {
      let stored = uploaded;

      if (!stored) {
        const uploadResult = await uploadFile(file).catch(async (firstError) => {
          if (requestId !== transcribeReqRef.current) {
            throw firstError;
          }
          console.warn("[analysis] first upload attempt failed, retrying once", {
            file: file.name,
            error: firstError,
          });
          setStatusMessage(`Server was still waking up — retrying ${file.name}...`);
          await new Promise((resolve) => window.setTimeout(resolve, 1500));
          return uploadFile(file);
        });
        stored = { id: uploadResult.id, filename: uploadResult.filename };
      }

      const created = await createTranscribeJob(file, stored).catch(async (firstError) => {
        if (requestId !== transcribeReqRef.current) {
          throw firstError;
        }
        console.warn("[analysis] first job-creation attempt failed, retrying once", {
          file: file.name,
          error: firstError,
        });
        setStatusMessage(`Server was still waking up — retrying ${file.name}...`);
        await new Promise((resolve) => window.setTimeout(resolve, 1500));
        return createTranscribeJob(file, stored);
      });
      window.clearTimeout(slowTimer);

      if (requestId !== transcribeReqRef.current) {
        return;
      }
      setStatusMessage(`Queued for transcription...`);

      const result = await pollTranscribeJob(created.job_id, {
        isSuperseded: () => requestId !== transcribeReqRef.current || !mountedRef.current,
        onStatusChange: (status, elapsedMs) => {
          if (requestId !== transcribeReqRef.current || !mountedRef.current) {
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
      });

      if (requestId !== transcribeReqRef.current || !mountedRef.current) {
        return;
      }

      setTranscription({
        ...result,
        sourceName: file.name,
        uploadedFilename: stored.filename,
        jobId: created.job_id,
      });
      setStatusMessage(`Transcription ready for ${file.name}.`);

      // generate-variants / choose-progression reference the upload by id.
      // The transcription summary is filled by runAnalysis right after (from
      // /api/analyze) -- the transcribe result no longer carries it.
      useSessionStore.getState().setLastUpload({
        uploadId: stored.id,
        filename: stored.filename,
        sourceName: file.name,
        transcription: { chords: [], key: "", moodLabel: "neutral", pitchHistogram: [] },
      });

      const defaultEnd = Math.min(DEFAULT_CLIP_SEC, Math.max(1, result.source_duration_sec));
      const window0: ClipWindow = { start: 0, end: defaultEnd };
      setClipWindow(window0);
      void runAnalysis(created.job_id, window0);
    } catch (transcribeError) {
      window.clearTimeout(slowTimer);
      if (transcribeError instanceof TranscribeJobSupersededError) {
        return;
      }
      console.error("[analysis] failed", { file: file.name, error: transcribeError });
      if (requestId === transcribeReqRef.current && mountedRef.current) {
        setErrorMessage(
          transcribeError instanceof Error ? transcribeError.message : "Transcription failed",
        );
        setStatusMessage("We couldn't transcribe that audio clip.");
      }
    } finally {
      if (requestId === transcribeReqRef.current && mountedRef.current) {
        setIsTranscribing(false);
      }
    }
  };

  const handleUploadSuccess = ({ id, filename, file }: UploadSuccessPayload) => {
    setSelectedFile(file);
    setErrorMessage(null);
    setStatusMessage(`Uploaded ${file.name}. Starting transcription...`);
    void runTranscription(file, { id, filename });
  };

  const handleRecordingComplete = (file: File) => {
    setShowRecorder(true);
    setSelectedFile(file);
    setErrorMessage(null);
    setStatusMessage(`Recording ready. Transcribing ${file.name}...`);
    void runTranscription(file, null);
  };

  const handleClipCommit = (next: ClipWindow) => {
    if (!transcription) return;
    setClipWindow(next);
    void runAnalysis(transcription.jobId, next);
  };

  const analysisData = analysis.status === "ready" ? analysis.data : null;
  const groupedChords = analysisData ? groupChordsByRoot(analysisData.detected_chords) : {};
  const mood = analysisData ? moodMeta[analysisData.mood_label] : null;

  return (
    <ErrorBoundary>
      <>
        <main className="space-y-8 pb-8">
          <section className="grid gap-6 xl:grid-cols-2">
            <div className="rounded-[1.8rem] border border-white/10 bg-[rgba(17,22,32,0.72)] p-6 shadow-[0_10px_24px_rgba(2,6,23,0.2)] backdrop-blur-md">
              <div className="mb-5 text-[11px] font-medium tracking-[0.22em] text-white/45 uppercase">
                Upload audio
              </div>

              <UploadButton
                onUploadSuccess={handleUploadSuccess}
                onUploadError={setErrorMessage}
                label="Upload Audio"
              />

              {selectedFile ? (
                <div className="mt-5 rounded-2xl border border-white/10 bg-[rgba(10,14,22,0.5)] p-4 text-sm text-[#dfe7f5]/70">
                  <p className="text-[11px] font-medium tracking-[0.22em] text-white/45 uppercase">
                    Source
                  </p>
                  <p className="mt-2 font-medium text-white">{selectedFile.name}</p>
                </div>
              ) : null}
            </div>

            <div className="rounded-[1.8rem] border border-white/10 bg-[rgba(17,22,32,0.72)] p-6 shadow-[0_10px_24px_rgba(2,6,23,0.2)] backdrop-blur-md">
              <div className="mb-5 text-[11px] font-medium tracking-[0.22em] text-white/45 uppercase">
                Record audio
              </div>

              <button
                type="button"
                onClick={() => setShowRecorder((current) => !current)}
                className="flex min-h-[220px] w-full flex-col items-center justify-center gap-4 rounded-[1.4rem] border border-dashed border-[#8b5cf6]/45 bg-[rgba(139,92,246,0.06)] px-6 py-8 text-center text-[#f1e9ff] transition hover:border-[#b18aff] hover:bg-[rgba(139,92,246,0.12)]"
                aria-expanded={showRecorder}
              >
                <span className="flex h-14 w-14 items-center justify-center rounded-full border border-[#d19af7]/50 bg-[#8b5cf6]/15 text-2xl text-[#d19af7] shadow-[0_0_18px_rgba(139,92,246,0.24)]">
                  ●
                </span>
                <span className="text-base font-semibold">
                  {showRecorder ? "Hide recorder" : "Record audio"}
                </span>
                <span className="text-sm text-white/45">
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
            </div>
          </section>

          <div className="space-y-6">
            {isTranscribing ? (
              <section className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/20 backdrop-blur-md">
                <TranscribingAnimation statusMessage={statusMessage} />
              </section>
            ) : null}

            {/* Transcription result -- the full-length MIDI + downloads. */}
            {transcription ? (
              <section className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/20 backdrop-blur-md">
                <div className="space-y-5">
                  <div>
                    <p
                      className="text-sm font-semibold text-white/75"
                      style={{ textShadow: "0 2px 8px rgba(0,0,0,0.8)" }}
                    >
                      Transcription
                    </p>
                    <h2
                      className="mt-1 text-2xl font-semibold text-white"
                      style={{ textShadow: "0 2px 8px rgba(0,0,0,0.8)" }}
                    >
                      {transcription.sourceName}
                    </h2>
                    <p className="mt-2 text-sm text-white/65">
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
                      className="rounded-full border border-sky-400/40 bg-sky-500/10 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:border-sky-300 hover:bg-sky-500/20"
                    >
                      View &amp; download result
                    </Link>
                  </div>
                </div>
              </section>
            ) : null}

            {/* Clip analysis -- re-runnable against any window of the source. */}
            {transcription ? (
              <section className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/20 backdrop-blur-md">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                  <div>
                    <p
                      className="text-sm font-semibold text-white/75"
                      style={{ textShadow: "0 2px 8px rgba(0,0,0,0.8)" }}
                    >
                      Clip analysis
                    </p>
                    <p className="mt-1 text-sm text-white/65">
                      Mood, key, tempo and chords for a section of the audio. Pick a window and
                      re-analyse.
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

                <div className="mt-5 rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm">
                  <ClipRange
                    sourceDurationSec={transcription.source_duration_sec}
                    value={clipWindow}
                    onCommit={handleClipCommit}
                    busy={analysis.status === "loading"}
                  />
                </div>

                <div aria-live="polite" className="mt-5">
                  {analysis.status === "loading" ? (
                    <div className="flex items-center justify-center py-10">
                      <Spinner label="Analysing this section..." />
                    </div>
                  ) : analysis.status === "error" ? (
                    <div className="rounded-2xl border border-red-500/40 bg-red-950/40 p-4 text-sm text-red-100">
                      {analysis.error}
                    </div>
                  ) : analysisData ? (
                    <div className="space-y-6">
                      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                        <div className="rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm">
                          <p className="text-xs tracking-[0.2em] text-white/45 uppercase">Key</p>
                          <p className="mt-3 text-lg font-semibold text-white">
                            🔑 {analysisData.key}
                          </p>
                        </div>
                        <div className="rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm">
                          <p className="text-xs tracking-[0.2em] text-white/45 uppercase">
                            Notes in window
                          </p>
                          <p className="mt-3 text-lg font-semibold text-white">
                            {analysisData.n_notes}
                          </p>
                        </div>
                        <div className="rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm">
                          <p className="text-xs tracking-[0.2em] text-white/45 uppercase">Tempo</p>
                          <p className="mt-3 text-lg font-semibold text-white">
                            {analysisData.tempo_bpm} BPM
                          </p>
                        </div>
                        <div className="rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm">
                          <p className="text-xs tracking-[0.2em] text-white/45 uppercase">
                            Average pitch
                          </p>
                          <p className="mt-3 text-lg font-semibold text-white">
                            {analysisData.average_pitch}
                          </p>
                        </div>
                      </div>

                      <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm">
                        <p
                          className="text-sm font-semibold text-white"
                          style={{ textShadow: "0 2px 8px rgba(0,0,0,0.8)" }}
                        >
                          Pitch histogram
                        </p>
                        <p className="mt-1 text-sm text-white/65">
                          Pitch-class balance across {formatDuration(analysisData.clip_start_sec)}–
                          {formatDuration(analysisData.clip_end_sec)}.
                        </p>
                        <div className="mt-5 grid grid-cols-12 gap-2">
                          {PITCH_CLASS_LABELS.map((label, index) => {
                            const value = analysisData.pitch_histogram[index] ?? 0;
                            return (
                              <div key={label} className="flex flex-col items-center gap-2">
                                <div className="flex h-28 w-full items-end rounded-2xl border border-white/10 bg-black/20 p-2">
                                  <div
                                    className="w-full rounded-xl bg-gradient-to-t from-purple-500 via-fuchsia-400 to-sky-300"
                                    style={{ height: `${Math.max(value * 100, 8)}%` }}
                                  />
                                </div>
                                <span className="text-[11px] text-white/60">{label}</span>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm">
                        <p
                          className="text-sm font-semibold text-white"
                          style={{ textShadow: "0 2px 8px rgba(0,0,0,0.8)" }}
                        >
                          Detected chords
                        </p>
                        <p className="mt-2 text-sm text-white/65">
                          Hover a chord to preview a guitar fingering. Chords are grouped by root
                          note.
                        </p>
                        <div className="mt-4 space-y-4">
                          {Object.entries(groupedChords).length > 0 ? (
                            Object.entries(groupedChords).map(([root, chords]) => (
                              <div key={root} className="space-y-2">
                                <p className="text-xs tracking-[0.2em] text-white/45 uppercase">
                                  {root}
                                </p>
                                <div className="flex flex-wrap gap-2">
                                  {chords.map((chord, index) => (
                                    <ChordDiagram key={`${chord}-${index}`} chord={chord} />
                                  ))}
                                </div>
                              </div>
                            ))
                          ) : (
                            <p className="text-sm text-white/60">
                              No chord labels were detected for this window.
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>
              </section>
            ) : null}
          </div>
        </main>

        {errorMessage ? <ErrorToast message={errorMessage} onDismiss={dismissError} /> : null}
      </>
    </ErrorBoundary>
  );
}
