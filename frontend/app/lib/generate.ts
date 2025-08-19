// frontend/app/lib/generate.ts
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

if (!BACKEND_URL) {
  throw new Error("NEXT_PUBLIC_BACKEND_URL environment variable is not set.");
}

export async function generateMelody(file: File) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BACKEND_URL}/generate`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Melody generation failed: ${errorText}`);
  }

  return await res.json();
}

