import { checkAudioFile } from "../../src/domain/audioFile";
import { getPublicBackendApiUrl } from "./backendUrl";
import { requestJson } from "./request";

export type UploadResponse = { id: string; filename: string };

// Uploaded audio can exceed Vercel's ~4.5MB serverless function body limit,
// so this goes straight to the backend instead of through the Next.js
// /api/upload proxy route.
export async function uploadFile(
  file: File,
  options: { signal?: AbortSignal } = {},
): Promise<UploadResponse> {
  const rejection = checkAudioFile(file);
  if (rejection) {
    throw new Error(rejection.message);
  }

  const formData = new FormData();
  formData.append("file", file);

  const data = await requestJson<{ id?: string; filename?: string }>(
    getPublicBackendApiUrl("/upload"),
    {
      method: "POST",
      body: formData,
      expectedContentType: "application/json",
      signal: options.signal,
    },
  );

  if (!data.filename || !data.id) {
    throw new Error("Upload failed");
  }

  return { id: data.id, filename: data.filename };
}
