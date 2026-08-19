"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { AudioRecorder } from "../components/audio-recorder";
import { ChordDiagram } from "../components/chord-diagram";
import ErrorBoundary from "../components/error-boundary";
import { ErrorToast } from "../components/error-toast";
import { UploadButton, type UploadSuccessPayload } from "../components/upload-button";
import { getPublicBackendApiUrl } from "./lib/backendUrl";
import { requestJson } from "./lib/request";
import { useSessionStore } from "./lib/session-store";
import { uploadFile } from "./lib/upload";

type TabKey = "upload" | "record";

type AnalysisResult = {
  n_notes: number;
  duration_sec: number;
  source_duration_sec: number;
  truncated: boolean;
  midi_b64: string;
  wav_b64: string | null;
  midi_filename: string;
  wav_filename: string;
  mood_label: "happy" | "sad" | "neutral";
  mood_idx: number;
  detected_chords: string[];
  key: string;
  pitch_histogram: number[];
  tempo_bpm: number;
  average_pitch: number;
  sourceName: string;
  uploadedFilename: string;
};

const PITCH_CLASS_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const moodMeta = {
  happy: {
    emoji: "😄",
    label: "happy",
    classes: "border-yellow-500 bg-yellow-900/40 text-yellow-100",
  },
  sad: {
    emoji: "😢",
    label: "sad",
    classes: "border-blue-500 bg-blue-900/40 text-blue-100",
  },
  neutral: {
    emoji: "😐",
    label: "neutral",
    classes: "border-gray-600 bg-gray-800 text-gray-100",
  },
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

function triggerBase64Download(filename: string, data: string, mimeType: string) {
  const anchor = document.createElement("a");
  anchor.href = `data:${mimeType};base64,${data}`;
  anchor.download = filename;
  anchor.click();
}

function createAudioObjectUrl(base64Audio: string, mimeType: string) {
  const bytes = Uint8Array.from(atob(base64Audio), (character) => character.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type: mimeType }));
}

// Uploaded audio can exceed Vercel's ~4.5MB serverless function body limit,
// so this goes straight to the backend instead of through the Next.js
// /api/transcribe proxy route (same reasoning as uploadFile in lib/upload.ts).
async function transcribeFile(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  return requestJson<Omit<AnalysisResult, "sourceName" | "uploadedFilename">>(
    getPublicBackendApiUrl("/transcribe"),
    {
      method: "POST",
      body: formData,
      expectedContentType: "application/json",
    },
  );
}

function AnalysisAnimation() {
  const heights = [40, 65, 50, 80, 45, 70, 35];
  const durations = ["0.7s", "0.85s", "0.65s", "0.9s", "0.75s", "0.8s", "0.6s"];
  return (
    <div className="flex flex-col items-center justify-center gap-6 py-14">
      <div className="flex h-20 items-end gap-[6px]">
        {heights.map((h, i) => (
          <div
            key={i}
            className="w-3 animate-bounce rounded-full bg-gradient-to-t from-purple-500 to-sky-400"
            style={{
              height: `${h}px`,
              animationDelay: `${i * 0.11}s`,
              animationDuration: durations[i],
            }}
          />
        ))}
      </div>
      <p className="animate-pulse text-sm font-semibold tracking-wide text-white/75">
        Analysing your audio…
      </p>
      <p className="max-w-xs text-center text-xs text-white/45">
        The first analysis after a period of inactivity can take up to a couple of minutes while
        the backend wakes up — later ones are much faster.
      </p>
    </div>
  );
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabKey>("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedSourceName, setSelectedSourceName] = useState<string | null>(null);
  const [selectedUploadFilename, setSelectedUploadFilename] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [analysisAudioUrl, setAnalysisAudioUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [statusMessage, setStatusMessage] = useState(
    "Choose a file or record a clip to start analysis.",
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [pendingSourceName, setPendingSourceName] = useState<string | null>(null);
  const requestIdRef = useRef(0);

  // Stable reference: ErrorToast's auto-dismiss effect depends on this
  // callback, so a fresh function identity on every render would reset its
  // timer before it ever fires.
  const dismissError = useCallback(() => setErrorMessage(null), []);

  useEffect(() => {
    if (!analysisResult?.wav_b64) {
      setAnalysisAudioUrl(null);
      return;
    }

    const url = createAudioObjectUrl(analysisResult.wav_b64, "audio/wav");
    setAnalysisAudioUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [analysisResult?.wav_b64]);

  const runSelectedAnalysis = async (
    file: File,
    uploaded: { id: string; filename: string } | null,
  ) => {
    const requestId = ++requestIdRef.current;
    setErrorMessage(null);
    setPendingSourceName(file.name);
    setIsAnalyzing(true);
    setStatusMessage(`Analysing ${file.name}...`);

    // The backend can be waking up from Render's free-tier idle sleep, in
    // which case the first analysis after a while can take a lot longer
    // than usual — let the user know rather than leaving a bare spinner.
    const slowAnalysisTimer = window.setTimeout(() => {
      if (requestId === requestIdRef.current) {
        setStatusMessage(
          `Still analysing ${file.name}... this can take a minute or two if the server was idle.`,
        );
      }
    }, 8000);

    try {
      let stored = uploaded;

      if (!stored) {
        const uploadResult = await uploadFile(file);
        stored = { id: uploadResult.id, filename: uploadResult.filename };
        setSelectedUploadFilename(uploadResult.filename);
      }

      // A cold backend can fail the first /transcribe call outright (Render's
      // free-tier gateway can time out before our own analysis finishes) even
      // though the model finishes loading server-side moments later — so a
      // retry right after a failure is effectively always fast. Retry once
      // before surfacing an error to the user.
      const result = await transcribeFile(file).catch(async (firstError) => {
        if (requestId !== requestIdRef.current) {
          throw firstError;
        }
        console.warn("[analysis] first transcribe attempt failed, retrying once", {
          file: file.name,
          error: firstError,
        });
        setStatusMessage(`Server was still waking up — retrying ${file.name}...`);
        await new Promise((resolve) => window.setTimeout(resolve, 1500));
        return transcribeFile(file);
      });
      window.clearTimeout(slowAnalysisTimer);

      if (requestId !== requestIdRef.current) {
        return;
      }

      setAnalysisResult({
        ...result,
        sourceName: file.name,
        uploadedFilename: stored.filename,
      });

      useSessionStore.getState().setLastUpload({
        uploadId: stored.id,
        filename: stored.filename,
        sourceName: file.name,
        transcription: {
          chords: result.detected_chords,
          key: result.key,
          moodLabel: result.mood_label,
          pitchHistogram: result.pitch_histogram,
        },
      });

      setStatusMessage(`Analysis ready for ${file.name}.`);
      setPendingSourceName(null);
    } catch (analysisError) {
      window.clearTimeout(slowAnalysisTimer);
      console.error("[analysis] failed", { file: file.name, error: analysisError });
      if (requestId === requestIdRef.current) {
        setErrorMessage(analysisError instanceof Error ? analysisError.message : "Analysis failed");
        setStatusMessage("We couldn't analyse that audio clip.");
      }
    } finally {
      if (requestId === requestIdRef.current) {
        setIsAnalyzing(false);
      }
    }
  };

  const handleUploadSuccess = ({ id, filename, file }: UploadSuccessPayload) => {
    setActiveTab("upload");
    setSelectedFile(file);
    setSelectedSourceName(file.name);
    setSelectedUploadFilename(filename);
    setErrorMessage(null);
    setStatusMessage(`Uploaded ${file.name}. Starting analysis...`);
    void runSelectedAnalysis(file, { id, filename });
  };

  const handleRecordingComplete = (file: File) => {
    setActiveTab("record");
    setSelectedFile(file);
    setSelectedSourceName(file.name);
    setSelectedUploadFilename(null);
    setErrorMessage(null);
    setStatusMessage(`Recording ready. Starting analysis for ${file.name}...`);
    void runSelectedAnalysis(file, null);
  };

  const groupedChords = analysisResult ? groupChordsByRoot(analysisResult.detected_chords) : {};
  const mood = analysisResult ? moodMeta[analysisResult.mood_label] : null;

  return (
    <ErrorBoundary>
      <>
        <main className="space-y-8 pb-8">
          <section className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
            <div className="space-y-6 rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/20 backdrop-blur-md">
              <div className="space-y-3">
                <UploadButton
                  onUploadSuccess={handleUploadSuccess}
                  onUploadError={setErrorMessage}
                  label="Upload Audio"
                />

                <button
                  type="button"
                  onClick={() => setActiveTab("record")}
                  className="inline-flex items-center justify-center overflow-hidden rounded-2xl border border-purple-400/40 bg-purple-600/20 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-purple-950/20 transition hover:border-purple-300 hover:bg-purple-500/25"
                >
                  Record Audio
                </button>

                {activeTab === "record" ? (
                  <AudioRecorder onRecordingComplete={handleRecordingComplete} showLivePitch />
                ) : null}
              </div>

              <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <p className="text-sm font-semibold text-white">Analyse source</p>
                    <p className="mt-2 text-sm text-white/70">
                      Analysis starts automatically after upload or recording.
                    </p>
                  </div>
                </div>

                {selectedFile ? (
                  <p className="mt-3 text-sm text-white/70">
                    Selected source:{" "}
                    <span className="font-medium text-white">{selectedFile.name}</span>
                  </p>
                ) : null}
              </div>
            </div>

            <div className="space-y-6">
              {isAnalyzing ? (
                <section className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/20 backdrop-blur-md">
                  <AnalysisAnimation />
                </section>
              ) : null}

              <section className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/20 backdrop-blur-md">
                {!analysisResult ? (
                  <div className="flex min-h-[320px] items-center justify-center rounded-3xl border border-dashed border-white/15 bg-white/5 p-8 text-center text-white/65 backdrop-blur-sm">
                    Upload or record audio to see the transcription and downloads.
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                      <div>
                        <p
                          className="text-sm font-semibold text-white/75"
                          style={{ textShadow: "0 2px 8px rgba(0,0,0,0.8)" }}
                        >
                          Results
                        </p>
                        <h2
                          className="mt-1 text-2xl font-semibold text-white"
                          style={{ textShadow: "0 2px 8px rgba(0,0,0,0.8)" }}
                        >
                          {analysisResult.sourceName}
                        </h2>
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

                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm">
                        <p className="text-xs tracking-[0.2em] text-white/45 uppercase">Key</p>
                        <p className="mt-3 text-lg font-semibold text-white">
                          🔑 {analysisResult.key}
                        </p>
                      </div>

                      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm">
                        <p className="text-xs tracking-[0.2em] text-white/45 uppercase">
                          Detected notes
                        </p>
                        <p className="mt-3 text-lg font-semibold text-white">
                          {analysisResult.n_notes}
                        </p>
                      </div>

                      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm">
                        <p className="text-xs tracking-[0.2em] text-white/45 uppercase">Tempo</p>
                        <p className="mt-3 text-lg font-semibold text-white">
                          {analysisResult.tempo_bpm} BPM
                        </p>
                      </div>

                      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm">
                        <p className="text-xs tracking-[0.2em] text-white/45 uppercase">
                          Average pitch
                        </p>
                        <p className="mt-3 text-lg font-semibold text-white">
                          {analysisResult.average_pitch}
                        </p>
                      </div>
                    </div>

                    <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <p
                            className="text-sm font-semibold text-white"
                            style={{ textShadow: "0 2px 8px rgba(0,0,0,0.8)" }}
                          >
                            Pitch histogram: Tonal summary of your audio input
                          </p>
                          <p className="mt-1 text-sm text-white/65">
                            Pitch-class balance across the transcription.
                          </p>
                        </div>

                        <span className="text-sm text-white/50">
                          {analysisResult.duration_sec.toFixed(2)} s
                        </span>
                      </div>

                      {analysisResult.truncated ? (
                        <p className="mt-3 rounded-2xl border border-amber-500/30 bg-amber-900/20 px-4 py-2 text-sm text-amber-100">
                          This clip is {formatDuration(analysisResult.source_duration_sec)} long —
                          only the first {formatDuration(analysisResult.duration_sec)} was analysed.
                        </p>
                      ) : null}

                      <div className="mt-5 grid grid-cols-12 gap-2">
                        {PITCH_CLASS_LABELS.map((label, index) => {
                          const value = analysisResult.pitch_histogram[index] ?? 0;

                          return (
                            <div key={label} className="flex flex-col items-center gap-2">
                              <div className="flex h-28 w-full items-end rounded-2xl border border-white/10 bg-black/20 p-2">
                                <div
                                  className="w-full rounded-xl bg-gradient-to-t from-purple-500 via-fuchsia-400 to-sky-300"
                                  style={{
                                    height: `${Math.max(value * 100, 8)}%`,
                                  }}
                                />
                              </div>
                              <span className="text-[11px] text-white/60">{label}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <div className="grid gap-6 lg:grid-cols-[1fr_0.9fr]">
                      <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm">
                        <div className="flex flex-wrap gap-3">
                          <button
                            type="button"
                            onClick={() =>
                              triggerBase64Download(
                                analysisResult.midi_filename,
                                analysisResult.midi_b64,
                                "audio/midi",
                              )
                            }
                            className="rounded-full border border-sky-400/40 bg-sky-500/10 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:border-sky-300 hover:bg-sky-500/20"
                          >
                            Download MIDI
                          </button>

                          {analysisResult.wav_b64 ? (
                            <button
                              type="button"
                              onClick={() =>
                                triggerBase64Download(
                                  analysisResult.wav_filename || "transcription.wav",
                                  analysisResult.wav_b64!,
                                  "audio/wav",
                                )
                              }
                              className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:border-emerald-300 hover:bg-emerald-500/20"
                            >
                              Download WAV
                            </button>
                          ) : null}
                        </div>

                        {analysisAudioUrl ? (
                          <audio controls className="mt-4 w-full" src={analysisAudioUrl}>
                            Your browser does not support the audio element.
                          </audio>
                        ) : (
                          <p className="mt-4 text-sm text-white/65">
                            WAV preview will appear here when FluidSynth is available on the
                            backend.
                          </p>
                        )}
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
                        Hover a chord to preview a guitar fingering diagram. Click on a chord to
                        listen. Chords are grouped by root note.
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
                            No chord labels were detected for this clip.
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </section>
            </div>
          </section>
        </main>

        {errorMessage ? <ErrorToast message={errorMessage} onDismiss={dismissError} /> : null}
      </>
    </ErrorBoundary>
  );
}
