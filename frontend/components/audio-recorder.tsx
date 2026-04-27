"use client";

import { useEffect, useRef, useState } from "react";
import { useAudioAnalyzer } from "../hooks/use-audio-analyzer";
import { LivePitchHistogram } from "./live-pitch-histogram";

type AudioRecorderProps = {
  onRecordingComplete: (file: File) => void;
  showLivePitch?: boolean;
};

function getSupportedMimeType() {
  if (typeof MediaRecorder === "undefined") {
    return "";
  }

  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/wav"];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) ?? "";
}

export function AudioRecorder({ onRecordingComplete, showLivePitch = false }: AudioRecorderProps) {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const previewUrlRef = useRef<string | null>(null);

  const [isRecording, setIsRecording] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [recordedFile, setRecordedFile] = useState<File | null>(null);
  const [status, setStatus] = useState("Ready to record");
  const [error, setError] = useState<string | null>(null);
  // Mirrors streamRef as React state so useAudioAnalyzer re-runs when it changes
  const [liveStream, setLiveStream] = useState<MediaStream | null>(null);

  const analyzerState = useAudioAnalyzer(liveStream, isRecording && showLivePitch);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  const stopStream = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  };

  const clearPreviewState = () => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setPreviewUrl(null);
    setRecordedFile(null);
  };

  const startRecording = async () => {
    setError(null);
    clearPreviewState();
    stopStream();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = getSupportedMimeType();
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

      chunksRef.current = [];
      streamRef.current = stream;
      mediaRecorderRef.current = recorder;
      setLiveStream(stream);

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blobType = recorder.mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: blobType });
        const extension = blobType.includes("wav") ? "wav" : "webm";
        const normalizedType = extension === "wav" ? "audio/wav" : "audio/webm";
        const file = new File([blob], `recording.${extension}`, { type: normalizedType });
        if (previewUrlRef.current) {
          URL.revokeObjectURL(previewUrlRef.current);
        }
        const nextPreviewUrl = URL.createObjectURL(blob);
        previewUrlRef.current = nextPreviewUrl;
        setPreviewUrl(nextPreviewUrl);
        setRecordedFile(file);
        setStatus("Preview ready. Use this recording or record again.");
        setLiveStream(null);
        stopStream();
        mediaRecorderRef.current = null;
      };

      recorder.start();
      setIsRecording(true);
      setStatus("Recording...");
    } catch (recordingError) {
      const message = recordingError instanceof Error ? recordingError.message : "Microphone access failed";
      setError(message);
      setStatus("Recording unavailable");
    }
  };

  const stopRecording = () => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      setIsRecording(false);
    }
  };

  const useRecording = () => {
    if (!recordedFile) {
      return;
    }
    setStatus("Recording selected.");
    onRecordingComplete(recordedFile);
  };

  const recordAgain = () => {
    if (isRecording) {
      stopRecording();
    }
    setLiveStream(null);
    clearPreviewState();
    setStatus("Ready to record");
    setError(null);
  };

  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-xl shadow-black/20 backdrop-blur-sm">
      <div className="flex flex-col gap-4">
        <div>
          <p className="text-sm font-semibold text-white">Browser recorder</p>
          <p className="mt-1 text-sm text-gray-400">
            
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={startRecording}
            disabled={isRecording}
            className="rounded-2xl border border-rose-400/40 bg-rose-500/15 px-4 py-3 text-sm font-semibold text-rose-100 transition hover:border-rose-300 hover:bg-rose-500/20 disabled:cursor-not-allowed disabled:opacity-60"
          >
            🎙 Record
          </button>
          <button
            type="button"
            onClick={stopRecording}
            disabled={!isRecording}
            className="rounded-2xl border border-sky-400/40 bg-sky-500/15 px-4 py-3 text-sm font-semibold text-sky-100 transition hover:border-sky-300 hover:bg-sky-500/20 disabled:cursor-not-allowed disabled:opacity-60"
          >
            Stop
          </button>
        </div>
        {showLivePitch && isRecording && (
          <LivePitchHistogram
            noteHistogram={analyzerState.noteHistogram}
            pitchClass={analyzerState.pitchClass}
            currentNote={analyzerState.currentNote}
            currentFrequency={analyzerState.currentFrequency}
            clarity={analyzerState.clarity}
          />
        )}

        <div className="rounded-2xl border border-white/10 bg-black/20 p-4 text-sm text-gray-300">
          <p>{status}</p>
          {error ? <p className="mt-2 text-red-300">{error}</p> : null}
          {previewUrl ? (
            <div className="mt-4 space-y-4">
              <audio controls className="w-full">
                <source src={previewUrl} />
                Your browser does not support the audio element.
              </audio>
              <div className="flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={useRecording}
                  className="rounded-2xl border border-emerald-400/40 bg-emerald-500/15 px-4 py-3 text-sm font-semibold text-emerald-100 transition hover:border-emerald-300 hover:bg-emerald-500/20"
                >
                  ✅ Use this recording
                </button>
                <button
                  type="button"
                  onClick={recordAgain}
                  className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm font-semibold text-gray-200 transition hover:border-white/20 hover:bg-white/10"
                >
                  🔄 Record again
                </button>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
