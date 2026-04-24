"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { AudioRecorder } from "../components/audio-recorder";
import { ChordDiagram } from "../components/chord-diagram";
import ErrorBoundary from "../components/error-boundary";
import { ErrorToast } from "../components/error-toast";
import {
  UploadButton,
  type UploadSuccessPayload,
} from "../components/upload-button";
import { requestJson } from "./lib/request";
import { uploadFile } from "./lib/upload";

type TabKey = "upload" | "record";

type AnalysisResult = {
  n_notes: number;
  duration_sec: number;
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

const PITCH_CLASS_LABELS = [
  "C",
  "C#",
  "D",
  "D#",
  "E",
  "F",
  "F#",
  "G",
  "G#",
  "A",
  "A#",
  "B",
];

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

function triggerBase64Download(
  filename: string,
  data: string,
  mimeType: string
) {
  const anchor = document.createElement("a");
  anchor.href = `data:${mimeType};base64,${data}`;
  anchor.download = filename;
  anchor.click();
}

function createAudioObjectUrl(base64Audio: string, mimeType: string) {
  const bytes = Uint8Array.from(atob(base64Audio), (character) =>
    character.charCodeAt(0)
  );
  return URL.createObjectURL(new Blob([bytes], { type: mimeType }));
}

async function transcribeFile(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  return requestJson<Omit<AnalysisResult, "sourceName" | "uploadedFilename">>(
    "/api/transcribe",
    {
      method: "POST",
      body: formData,
      expectedContentType: "application/json",
    }
  );
}

function LoadingSkeleton() {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {[0, 1, 2, 3].map((index) => (
        <div
          key={index}
          className="h-28 animate-pulse rounded-3xl border border-white/10 bg-white/5 backdrop-blur-sm"
        />
      ))}
    </div>
  );
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabKey>("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedSourceName, setSelectedSourceName] = useState<string | null>(
    null
  );
  const [selectedUploadFilename, setSelectedUploadFilename] = useState<
    string | null
  >(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null
  );
  const [analysisAudioUrl, setAnalysisAudioUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [statusMessage, setStatusMessage] = useState(
    "Choose a file or record a clip to start analysis."
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [pendingSourceName, setPendingSourceName] = useState<string | null>(
    null
  );
  const requestIdRef = useRef(0);

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
    uploadedFilename: string | null
  ) => {
    const requestId = ++requestIdRef.current;
    setErrorMessage(null);
    setPendingSourceName(file.name);
    setIsAnalyzing(true);
    setStatusMessage(`Analysing ${file.name}...`);

    try {
      let storedFilename = uploadedFilename;

      if (!storedFilename) {
        const uploadResult = await uploadFile(file);
        storedFilename = uploadResult.filename;
        setSelectedUploadFilename(uploadResult.filename);
      }

      const result = await transcribeFile(file);

      if (requestId !== requestIdRef.current) {
        return;
      }

      setAnalysisResult({
        ...result,
        sourceName: file.name,
        uploadedFilename: storedFilename,
      });

      setStatusMessage(`Analysis ready for ${file.name}.`);
      setPendingSourceName(null);
    } catch (analysisError) {
      if (requestId === requestIdRef.current) {
        setErrorMessage(
          analysisError instanceof Error
            ? analysisError.message
            : "Analysis failed"
        );
        setStatusMessage("We couldn't analyse that audio clip.");
      }
    } finally {
      if (requestId === requestIdRef.current) {
        setIsAnalyzing(false);
      }
    }
  };

  const handleUploadSuccess = ({ filename, file }: UploadSuccessPayload) => {
    setActiveTab("upload");
    setSelectedFile(file);
    setSelectedSourceName(file.name);
    setSelectedUploadFilename(filename);
    setErrorMessage(null);
    setStatusMessage(`Uploaded ${file.name}. Starting analysis...`);
    void runSelectedAnalysis(file, filename);
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

  const resetForNewInput = (nextTab: TabKey) => {
    setActiveTab(nextTab);
    setSelectedFile(null);
    setSelectedSourceName(null);
    setSelectedUploadFilename(null);
    setPendingSourceName(null);
    setErrorMessage(null);
    setStatusMessage(
      nextTab === "upload"
        ? "Choose a new file to start analysis."
        : "Record a fresh idea and we'll analyse it automatically."
    );
  };

  const groupedChords = analysisResult
    ? groupChordsByRoot(analysisResult.detected_chords)
    : {};
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
                  <AudioRecorder onRecordingComplete={handleRecordingComplete} />
                ) : null}
              </div>

              <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <p className="text-sm font-semibold text-white">
                      Analyse source
                    </p>
                    <p className="mt-2 text-sm text-white/70">
                      Analysis starts automatically after upload or recording.
                    </p>
                  </div>
                </div>

                {selectedFile ? (
                  <p className="mt-3 text-sm text-white/70">
                    Selected source:{" "}
                    <span className="font-medium text-white">
                      {selectedFile.name}
                    </span>
                  </p>
                ) : null}
              </div>

              {analysisResult ? (
                <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm">
                  <p className="text-sm font-semibold text-white">Next step</p>
                  <p className="mt-2 text-sm text-white/70">
                    Explore CVAE + IDDM-PPO generations on the dedicated variants
                    page.
                  </p>
                  <div className="mt-4 flex flex-wrap gap-3">
                    <button
                      type="button"
                      onClick={() => resetForNewInput("upload")}
                      className="rounded-full border border-white/15 bg-white/5 px-4 py-2 text-sm font-semibold text-white transition hover:border-white/30 hover:bg-white/10"
                    >
                      Upload New Audio
                    </button>

                    <button
                      type="button"
                      onClick={() => resetForNewInput("record")}
                      className="rounded-full border border-white/15 bg-white/5 px-4 py-2 text-sm font-semibold text-white transition hover:border-white/30 hover:bg-white/10"
                    >
                      Record New Audio
                    </button>

                    <Link
                      href="/generate-variants"
                      className="rounded-full border border-emerald-400/40 bg-emerald-500/15 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:border-emerald-300 hover:bg-emerald-500/20"
                    >
                      Generate Variants
                    </Link>
                  </div>
                </div>
              ) : null}
            </div>

            <div className="space-y-6">
              {isAnalyzing ? (
                <section className="rounded-[2rem] border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/20 backdrop-blur-md">
                  <p className="mb-4 text-sm font-semibold text-white">
                    Analysis in progress
                  </p>
                  <LoadingSkeleton />
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
                        <p className="mt-2 text-sm text-white/60">
                          Stored as {analysisResult.uploadedFilename}
                        </p>
                      </div>

                      {mood ? (
                        <div
                          className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-semibold ${mood.classes}`}
                        >
                          <span>{mood.emoji}</span>
                          <span>{mood.label}</span>
                        </div>
                      ) : null}
                    </div>

                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm">
                        <p className="text-xs uppercase tracking-[0.2em] text-white/45">
                          Key
                        </p>
                        <p className="mt-3 text-lg font-semibold text-white">
                          🔑 {analysisResult.key}
                        </p>
                      </div>

                      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm">
                        <p className="text-xs uppercase tracking-[0.2em] text-white/45">
                          Detected notes
                        </p>
                        <p className="mt-3 text-lg font-semibold text-white">
                          {analysisResult.n_notes}
                        </p>
                      </div>

                      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm">
                        <p className="text-xs uppercase tracking-[0.2em] text-white/45">
                          Tempo
                        </p>
                        <p className="mt-3 text-lg font-semibold text-white">
                          {analysisResult.tempo_bpm} BPM
                        </p>
                      </div>

                      <div className="rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm">
                        <p className="text-xs uppercase tracking-[0.2em] text-white/45">
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

                      <div className="mt-5 grid grid-cols-12 gap-2">
                        {PITCH_CLASS_LABELS.map((label, index) => {
                          const value =
                            analysisResult.pitch_histogram[index] ?? 0;

                          return (
                            <div
                              key={label}
                              className="flex flex-col items-center gap-2"
                            >
                              <div className="flex h-28 w-full items-end rounded-2xl border border-white/10 bg-black/20 p-2">
                                <div
                                  className="w-full rounded-xl bg-gradient-to-t from-purple-500 via-fuchsia-400 to-sky-300"
                                  style={{
                                    height: `${Math.max(value * 100, 8)}%`,
                                  }}
                                />
                              </div>
                              <span className="text-[11px] text-white/60">
                                {label}
                              </span>
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
                                "audio/midi"
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
                                  analysisResult.wav_filename ||
                                    "transcription.wav",
                                  analysisResult.wav_b64,
                                  "audio/wav"
                                )
                              }
                              className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:border-emerald-300 hover:bg-emerald-500/20"
                            >
                              Download WAV
                            </button>
                          ) : null}
                        </div>

                        {analysisAudioUrl ? (
                          <audio
                            controls
                            className="mt-4 w-full"
                            src={analysisAudioUrl}
                          >
                            Your browser does not support the audio element.
                          </audio>
                        ) : (
                          <p className="mt-4 text-sm text-white/65">
                            WAV preview will appear here when FluidSynth is
                            available on the backend.
                          </p>
                        )}
                      </div>

                      <div className="rounded-3xl border border-white/10 bg-white/5 p-5 backdrop-blur-sm">
                        <div className="mt-4 grid gap-3 sm:grid-cols-2">
                          <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
                            <p className="text-xs uppercase tracking-[0.2em] text-white/45">
                              Mood
                            </p>
                            <p className="mt-2 text-lg font-semibold text-white">
                              {analysisResult.mood_label}
                            </p>
                          </div>


                        </div>
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
                        Hover a chord to preview a guitar fingering diagram.
                        Chords are grouped by root note.
                      </p>

                      <div className="mt-4 space-y-4">
                        {Object.entries(groupedChords).length > 0 ? (
                          Object.entries(groupedChords).map(([root, chords]) => (
                            <div key={root} className="space-y-2">
                              <p className="text-xs uppercase tracking-[0.2em] text-white/45">
                                {root}
                              </p>
                              <div className="flex flex-wrap gap-2">
                                {chords.map((chord, index) => (
                                  <ChordDiagram
                                    key={`${chord}-${index}`}
                                    chord={chord}
                                  />
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

        {errorMessage ? (
          <ErrorToast
            message={errorMessage}
            onDismiss={() => setErrorMessage(null)}
          />
        ) : null}
      </>
    </ErrorBoundary>
  );
}
