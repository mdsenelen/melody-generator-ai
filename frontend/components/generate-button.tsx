"use client";

import { useState } from "react";

import { requestJson } from "../app/lib/request";

export type GeneratedAudioPayload = {
  audio_b64: string;
  filename: string;
  download_path: string;
  detected_chords?: string[];
};

type GenerateButtonProps = {
  filename: string;
  onGenerated: (generated: GeneratedAudioPayload) => void;
  onError?: (message: string) => void;
  label?: string;
};

export function GenerateButton({
  filename,
  onGenerated,
  onError,
  label = "Generate accompaniment",
}: GenerateButtonProps) {
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const data = await requestJson<GeneratedAudioPayload>("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename }),
        expectedContentType: "application/json",
      });
      if (!data.audio_b64 || !data.filename) {
        throw new Error("Generation failed");
      }
      onGenerated(data);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Generation failed";
      onError?.(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <button
      onClick={handleGenerate}
      disabled={loading}
      className="rounded-2xl border border-emerald-400/40 bg-emerald-500/15 px-4 py-3 text-sm font-semibold text-emerald-100 transition hover:border-emerald-300 hover:bg-emerald-500/20 disabled:cursor-not-allowed disabled:opacity-60"
    >
      {loading ? "Generating..." : label}
    </button>
  );
}
