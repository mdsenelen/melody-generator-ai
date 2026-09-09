"use client";

import { useState } from "react";

import { uploadFile } from "../app/lib/upload";
import { Spinner } from "./spinner";

export type UploadSuccessPayload = {
  id: string;
  filename: string;
  file: File;
};

type UploadButtonProps = {
  onUploadSuccess: (payload: UploadSuccessPayload) => void;
  onUploadError?: (message: string) => void;
  label?: string;
};

export function UploadButton({
  onUploadSuccess,
  onUploadError,
  label = "Upload audio file",
}: UploadButtonProps) {
  const [status, setStatus] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }

    setIsLoading(true);

    try {
      const { id, filename } = await uploadFile(file);
      onUploadSuccess({ id, filename, file });
      setStatus("Upload complete");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Upload failed";
      setStatus(message);
      onUploadError?.(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-start gap-4">
      <label className="group relative flex min-h-[220px] w-full cursor-pointer flex-col items-center justify-center overflow-hidden rounded-[1.4rem] border border-dashed border-white/20 bg-[rgba(17,22,32,0.4)] px-6 py-8 text-center transition hover:border-[#8b5cf6]/70 hover:bg-[rgba(22,27,39,0.7)]">
        <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full border border-[#8b5cf6]/45 bg-[#8b5cf6]/10 text-xl text-[#d8b6ff] shadow-[0_0_12px_rgba(139,92,246,0.25)]">
          ↑
        </div>

        <span className="mb-2 text-[1.05rem] font-semibold text-[#f2f3f7]">
          {isLoading ? "Uploading..." : "Drop audio file here"}
        </span>
        <span className="text-sm text-[#dfe7f5]/70">
          {isLoading ? "Please wait" : "MP3 · WAV · FLAC · M4A · up to 50 MB"}
        </span>

        {isLoading ? <Spinner size="sm" label="Uploading" className="mt-4" /> : null}

        <input
          type="file"
          aria-label="Upload audio file"
          accept=".wav,.mp3,.flac,.ogg,.m4a,.webm,audio/*"
          onChange={handleFile}
          className="absolute inset-0 cursor-pointer opacity-0"
          disabled={isLoading}
        />
      </label>

      <div className="w-full">
        <p className="mb-2 text-[11px] font-medium tracking-[0.22em] text-white/45 uppercase">
          {label.toUpperCase()}
        </p>
        <p className="min-h-[1.25rem] text-sm text-[#dfe7f5]/60" aria-live="polite">
          {status || "Upload complete"}
        </p>
      </div>
    </div>
  );
}
