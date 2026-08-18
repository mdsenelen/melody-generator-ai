import { requestJson } from "./request";

export type UploadResponse = { id: string; filename: string };

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const data = await requestJson<{ id?: string; filename?: string }>("/api/upload", {
    method: "POST",
    body: formData,
    expectedContentType: "application/json",
  });

  if (!data.filename || !data.id) {
    throw new Error("Upload failed");
  }

  return { id: data.id, filename: data.filename };
}
