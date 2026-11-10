import * as z from "zod/mini";

// POST /api/upload (backend/app/main.py upload_audio): 201 with the stored
// upload's id and on-disk filename. Both are sent back verbatim later
// (upload_id / filename form fields), so neither may be empty.
export const uploadResponseSchema = z.object({
  id: z.string().check(z.minLength(1)),
  filename: z.string().check(z.minLength(1)),
});

export type UploadResponse = z.infer<typeof uploadResponseSchema>;
