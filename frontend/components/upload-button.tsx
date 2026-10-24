"use client";

import { useEffect, useRef, useState } from "react";

import { RetryCancelledError } from "../app/lib/retryOnWakeup";
import { uploadWhenAwake, WAKING_STATUS_MESSAGE } from "../app/lib/uploadWhenAwake";
import { Spinner } from "./spinner";

export type UploadSuccessPayload = {
  id: string;
  filename: string;
  file: File;
  // performance.now() when the file was chosen, before this upload -- lets
  // the caller time pick -> "Transcription ready" end to end.
  pickedAt: number;
};

// `idle` carries the transient "Upload complete" note; `error` keeps the
// message in the (non-red) status line until the next pick. While busy, the
// spinner (role="status") carries the text, so the line below stays empty
// rather than announcing the same thing twice.
type UploadState =
  | { status: "idle"; note: string }
  | { status: "waking" }
  | { status: "uploading" }
  | { status: "error"; message: string };

function statusText(state: UploadState) {
  switch (state.status) {
    case "idle":
      return state.note;
    case "waking":
    case "uploading":
      return "";
    case "error":
      return state.message;
  }
}

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
  const [state, setState] = useState<UploadState>({ status: "idle", note: "" });
  const isBusy = state.status === "waking" || state.status === "uploading";
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
    const pickedAt = performance.now();
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    if (statusTimerRef.current) {
      clearTimeout(statusTimerRef.current);
    }
    setState({ status: "uploading" });

    try {
      const { id, filename } = await uploadWhenAwake(file, {
        signal: controller.signal,
        onPhase: (phase) => {
          if (!controller.signal.aborted) {
            setState({ status: phase });
          }
        },
      });
      setState({ status: "idle", note: "Upload complete" });
      onUploadSuccess({ id, filename, file, pickedAt });
      statusTimerRef.current = setTimeout(() => setState({ status: "idle", note: "" }), 2000);
    } catch (error) {
      if (error instanceof RetryCancelledError || controller.signal.aborted) {
        return;
      }
      const message = error instanceof Error ? error.message : "Upload failed";
      setState({ status: "error", message });
      onUploadError?.(message);
    }
  };

  return (
    <div className="flex flex-col items-start gap-4">
      <label className="group border-border hover:border-border-strong relative flex min-h-[220px] w-full cursor-pointer flex-col items-center justify-center overflow-hidden rounded-[var(--radius)] border-2 border-dashed bg-transparent px-6 py-8 text-center transition-colors">
        <div className="border-primary/40 bg-primary/10 text-primary mb-4 flex h-12 w-12 items-center justify-center rounded-full border text-xl">
          ↑
        </div>

        {!isBusy ? (
          <>
            <span className="font-display text-foreground mb-2 text-[1.05rem]">
              Drop audio file here
            </span>
            <span className="text-secondary-foreground text-sm">
              MP3 · WAV · FLAC · M4A · up to 50 MB
            </span>
          </>
        ) : null}

        {isBusy ? (
          <Spinner
            size="sm"
            label={state.status === "waking" ? WAKING_STATUS_MESSAGE : "Uploading"}
            className="mt-4"
          />
        ) : null}

        <input
          type="file"
          aria-label="Upload audio file"
          accept=".wav,.mp3,.flac,.ogg,.m4a,.webm,audio/*"
          onChange={handleFile}
          className="absolute inset-0 cursor-pointer opacity-0"
          disabled={isBusy}
        />
      </label>

      <div className="w-full">
        <p className="text-muted-foreground mb-2 font-mono text-[11px] tracking-widest uppercase">
          {label.toUpperCase()}
        </p>
        <p className="text-muted-foreground min-h-[1.25rem] text-sm" aria-live="polite">
          {statusText(state)}
        </p>
      </div>
    </div>
  );
}
