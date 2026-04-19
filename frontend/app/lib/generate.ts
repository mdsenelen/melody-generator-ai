// frontend/app/lib/generate.ts
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

if (!BACKEND_URL) {
  throw new Error("NEXT_PUBLIC_BACKEND_URL environment variable is not set.");
}

async function requestJson<T>(input: RequestInfo | URL, init?: RequestInit) {
  const response = await fetch(input, init);
  const rawBody = await response.text();

  let parsed: T | null = null;
  try {
    parsed = JSON.parse(rawBody) as T;
  } catch {
    parsed = null;
  }

  if (!response.ok) {
    throw new Error(rawBody || "Request failed");
  }

  if (parsed === null) {
    throw new Error(`Expected JSON response but received: ${rawBody}`);
  }

  return parsed;
}

export async function generateMelody(file: File) {
  const form = new FormData();
  form.append("file", file);

  return requestJson(`${BACKEND_URL}/generate`, {
    method: "POST",
    body: form,
  });
}

