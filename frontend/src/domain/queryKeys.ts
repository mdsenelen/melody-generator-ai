// Every TanStack Query key in the app. Object-based so a key reads as data
// (and partial matches like `[{ scope: "clipAnalysis" }]` work), never a
// concatenated string.

export const queryKeys = {
  /** GET /api/transcribe/{jobId}: the status and result of a job of every kind. */
  job: (jobId: string) => [{ scope: "job", jobId }] as const,
  /** POST /api/analyze for one clip window of a completed transcription. */
  clipAnalysis: (jobId: string, startSec: number, endSec: number) =>
    [{ scope: "clipAnalysis", jobId, startSec, endSec }] as const,
  /** The disabled clip-analysis query, while there's no transcription to analyse. */
  noClipAnalysis: () => [{ scope: "clipAnalysis", jobId: null }] as const,
  /** GET /api/progressions: the curated library. */
  progressions: () => [{ scope: "progressions" }] as const,
};
