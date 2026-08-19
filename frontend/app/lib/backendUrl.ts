const DEFAULT_BACKEND_BASE_URL = "http://127.0.0.1:8000";

function isPlaceholderBackendUrl(value: string) {
  return /your-backend-url\.onrender\.com/i.test(value);
}

// Client-safe: only reads the NEXT_PUBLIC_ var, which Next.js inlines into
// the browser bundle. Server-only vars (BACKEND_BASE_URL, BACKEND_URL) are
// intentionally not read here since they aren't available client-side.
export function getPublicBackendApiUrl(path: string) {
  const configured = process.env.NEXT_PUBLIC_BACKEND_URL ?? DEFAULT_BACKEND_BASE_URL;
  const rawBaseUrl = isPlaceholderBackendUrl(configured) ? DEFAULT_BACKEND_BASE_URL : configured;
  const baseUrl = rawBaseUrl.replace(/\/+$/, "").replace(/\/api$/, "");
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${baseUrl}/api${normalizedPath}`;
}
