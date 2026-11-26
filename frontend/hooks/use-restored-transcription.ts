"use client";

import { skipToken, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { useSessionStore, type UploadSession } from "../app/lib/session-store";
import { getJob } from "../app/lib/transcribeJob";
import type { TranscriptionResult } from "../src/domain/contracts/job";
import { queryKeys } from "../src/domain/queryKeys";

export type RestoredTranscription = {
  jobId: string;
  sourceName: string;
  uploadedFilename: string;
  clipWindow: { start: number; end: number } | null;
  result: TranscriptionResult;
};

/**
 * The transcription this browser tab analysed earlier, looked up again from
 * its job id, so leaving /analyse and coming back doesn't lose it. The job
 * lives on the backend (and in the query cache shared with /result/[id]),
 * so only the id and the chosen window sit in sessionStorage -- never the
 * audio or the notes. Null while loading, when nothing was analysed, or when
 * the job has expired: a quiet empty page, not an error.
 *
 * The id is read after mount, not during render, so the first client render
 * matches the server's.
 */
export function useRestoredTranscription(): RestoredTranscription | null {
  const [saved, setSaved] = useState<UploadSession | null>(null);
  useEffect(() => {
    const last = useSessionStore.getState().lastUpload;
    setSaved(last?.jobId ? last : null);
  }, []);

  const jobId = saved?.jobId ?? null;
  const query = useQuery({
    queryKey: jobId ? queryKeys.job(jobId) : queryKeys.noJob(),
    queryFn: jobId ? ({ signal }) => getJob(jobId, { signal }) : skipToken,
  });

  const job = query.data;
  if (!saved || !jobId || job?.status !== "completed" || job.result?.kind !== "transcription") {
    return null;
  }
  return {
    jobId,
    sourceName: saved.sourceName,
    uploadedFilename: saved.filename,
    clipWindow: saved.clipWindow ?? null,
    result: job.result.data,
  };
}
