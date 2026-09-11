"use client";

import { useEffect, useRef, useState } from "react";

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
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (statusTimerRef.current) {
        clearTimeout(statusTimerRef.current);
      }
    };
  }, []);

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
      if (statusTimerRef.current) {
        clearTimeout(statusTimerRef.current);
      }
      statusTimerRef.current = setTimeout(() => setStatus(""), 2000);
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
      <label className="group border-border hover:border-border-strong relative flex min-h-[220px] w-full cursor-pointer flex-col items-center justify-center overflow-hidden rounded-[var(--radius)] border-2 border-dashed bg-transparent px-6 py-8 text-center transition-colors">
        <div className="border-primary/40 bg-primary/10 text-primary mb-4 flex h-12 w-12 items-center justify-center rounded-full border text-xl">
          ↑
        </div>

        {!isLoading ? (
          <>
            <span className="font-display text-foreground mb-2 text-[1.05rem]">
              Drop audio file here
            </span>
            <span className="text-secondary-foreground text-sm">
              MP3 · WAV · FLAC · M4A · up to 50 MB
            </span>
          </>
        ) : null}

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
        <p className="text-muted-foreground mb-2 font-mono text-[11px] tracking-widest uppercase">
          {label.toUpperCase()}
        </p>
        <p className="text-muted-foreground min-h-[1.25rem] text-sm" aria-live="polite">
          {status}
        </p>
      </div>
    </div>
  );
}
