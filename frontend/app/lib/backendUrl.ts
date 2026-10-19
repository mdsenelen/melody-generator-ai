const DEFAULT_BACKEND_BASE_URL = "http://127.0.0.1:8000";

function isPlaceholderBackendUrl(value: string) {
  return /your-backend-url\.onrender\.com/i.test(value);
}

// Client-safe: only reads the NEXT_PUBLIC_ var, which Next.js inlines into
// the browser bundle. Server-only vars (BACKEND_BASE_URL, BACKEND_URL) are
// intentionally not read here since they aren't available client-side.
function getPublicBackendOrigin() {
  const trimmed = process.env.NEXT_PUBLIC_BACKEND_URL?.trim();
  const configured = trimmed ? trimmed : DEFAULT_BACKEND_BASE_URL;
  const rawBaseUrl = isPlaceholderBackendUrl(configured) ? DEFAULT_BACKEND_BASE_URL : configured;
  return rawBaseUrl.replace(/\/+$/, "").replace(/\/api$/, "");
}

function normalizePath(path: string) {
  return path.startsWith("/") ? path : `/${path}`;
}

export function getPublicBackendApiUrl(path: string) {
  return `${getPublicBackendOrigin()}/api${normalizePath(path)}`;
}

// For the few routes registered on the FastAPI app directly rather than
// under /api (e.g. /health -- see backend/app/main.py).
export function getPublicBackendBaseUrl(path: string) {
  return `${getPublicBackendOrigin()}${normalizePath(path)}`;
}
