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
    <div className="flex flex-col items-start gap-3">
      <label className="relative inline-flex cursor-pointer items-center justify-center overflow-hidden rounded-2xl border border-purple-400/40 bg-purple-600/20 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-purple-950/20 transition hover:border-purple-300 hover:bg-purple-500/25">
        {isLoading ? <Spinner size="sm" label="Uploading" /> : <span>{label}</span>}
        <input
          type="file"
          accept=".wav,.mp3,.flac,.ogg,.m4a,.webm,audio/*"
          onChange={handleFile}
          className="absolute inset-0 cursor-pointer opacity-0"
          disabled={isLoading}
        />
      </label>
      <p className="min-h-[1.25rem] text-sm text-gray-300" aria-live="polite">
        {status}
      </p>
    </div>
  );
}
