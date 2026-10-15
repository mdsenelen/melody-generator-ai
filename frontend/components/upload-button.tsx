"use client";

import { useEffect, useRef, useState } from "react";

import { uploadFile } from "../app/lib/upload";
import { retryOnWakeup, RetryCancelledError } from "../app/lib/retryOnWakeup";
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
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      if (statusTimerRef.current) {
        clearTimeout(statusTimerRef.current);
      }
      abortControllerRef.current?.abort();
    };
  }, []);

  const handleFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);

    try {
      const { id, filename } = await retryOnWakeup((signal) => uploadFile(file, { signal }), {
        signal: controller.signal,
        onRetrying: () => setStatus("Server is waking up — retrying..."),
      });
      onUploadSuccess({ id, filename, file });
      setStatus("Upload complete");
      if (statusTimerRef.current) {
        clearTimeout(statusTimerRef.current);
      }
      statusTimerRef.current = setTimeout(() => setStatus(""), 2000);
    } catch (error) {
      if (error instanceof RetryCancelledError) {
        return;
      }
      const message = error instanceof Error ? error.message : "Upload failed";
      setStatus(message);
      onUploadError?.(message);
    } finally {
      if (!controller.signal.aborted) {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="flex flex-col items-start gap-2">
      <label className="group border-border-strong bg-raised/40 hover:border-primary focus-within:ring-ring relative flex min-h-[220px] w-full cursor-pointer flex-col items-center justify-center overflow-hidden rounded-[var(--radius)] border border-dashed px-6 py-8 text-center transition-colors focus-within:ring-2">
        <div
          aria-hidden="true"
          className="border-border-strong bg-background text-primary-bright mb-4 flex h-12 w-12 items-center justify-center rounded-full border text-xl shadow-[var(--shadow-knob)]"
        >
          ↑
        </div>

        {!isLoading ? (
          <>
            <span className="font-title text-foreground mb-2 text-lg">Drop audio file here</span>
            <span className="text-secondary-foreground text-sm">
              MP3 · WAV · FLAC · M4A · up to 50 MB
            </span>
          </>
        ) : null}

        {isLoading ? <Spinner size="sm" label="Uploading" className="mt-4" /> : null}

        <input
          type="file"
          aria-label={label}
          accept=".wav,.mp3,.flac,.ogg,.m4a,.webm,audio/*"
          onChange={handleFile}
          className="absolute inset-0 cursor-pointer opacity-0"
          disabled={isLoading}
        />
      </label>

      <div className="w-full">
        <p className="text-muted-foreground min-h-[1.25rem] text-sm" aria-live="polite">
          {status}
        </p>
      </div>
    </div>
  );
}
