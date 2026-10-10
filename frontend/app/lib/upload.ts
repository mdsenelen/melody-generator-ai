import { getPublicBackendApiUrl } from "./backendUrl";
import { requestJson } from "./request";

// Mirrors the backend's MAX_UPLOAD_BYTES default (50MB).
export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

export type UploadResponse = { id: string; filename: string };

// Uploaded audio can exceed Vercel's ~4.5MB serverless function body limit,
// so this goes straight to the backend instead of through the Next.js
// /api/upload proxy route.
export async function uploadFile(file: File): Promise<UploadResponse> {
  if (file.size > MAX_UPLOAD_BYTES) {
    throw new Error(`File is too large. The limit is ${MAX_UPLOAD_BYTES / (1024 * 1024)} MB.`);
  }

  const formData = new FormData();
  formData.append("file", file);

  const data = await requestJson<{ id?: string; filename?: string }>(
    getPublicBackendApiUrl("/upload"),
    {
      method: "POST",
      body: formData,
      expectedContentType: "application/json",
    },
  );

  if (!data.filename || !data.id) {
    throw new Error("Upload failed");
  }

  return { id: data.id, filename: data.filename };
}
