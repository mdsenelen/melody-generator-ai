// What the server will accept as an audio upload, checked before any bytes
// leave the browser. The server re-validates size (MAX_UPLOAD_BYTES
// middleware) and extension (POST /api/upload); this is the fast, friendly
// copy of those rules plus two the server can't phrase well: an empty file,
// and a non-audio file renamed to look like one.

/** Mirrors the backend's MAX_UPLOAD_BYTES default. */
export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

/** Mirrors backend/app/inference.py SUPPORTED_AUDIO_EXTENSIONS. */
export const AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"] as const;

// Audio-only files in a container browsers often label as video.
const AUDIO_CONTAINER_TYPES = new Set(["video/webm", "video/mp4", "video/ogg"]);

export type AudioFileRejection = {
  reason: "empty" | "unsupported_extension" | "unsupported_type" | "too_large";
  message: string;
};

type FileLike = Pick<File, "name" | "type" | "size">;

function hasAudioExtension(name: string): boolean {
  const lower = name.toLowerCase();
  return AUDIO_EXTENSIONS.some((extension) => lower.endsWith(extension));
}

// An empty type means the browser couldn't tell; the extension decides then.
function hasAudioType(type: string): boolean {
  return type === "" || type.startsWith("audio/") || AUDIO_CONTAINER_TYPES.has(type);
}

/** Null when the file can be uploaded; otherwise the first reason it can't. */
export function checkAudioFile(file: FileLike): AudioFileRejection | null {
  if (file.size === 0) {
    return {
      reason: "empty",
      message: `“${file.name}” is empty (0 bytes). Choose a file with audio in it.`,
    };
  }
  if (!hasAudioExtension(file.name)) {
    return {
      reason: "unsupported_extension",
      message: `“${file.name}” isn't a supported audio format. Use WAV, MP3, FLAC, OGG, M4A or WebM.`,
    };
  }
  if (!hasAudioType(file.type)) {
    return {
      reason: "unsupported_type",
      message: `“${file.name}” isn't an audio file (it's ${file.type}). Choose an audio recording.`,
    };
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return {
      reason: "too_large",
      message: `File is too large. The limit is ${MAX_UPLOAD_BYTES / (1024 * 1024)} MB.`,
    };
  }
  return null;
}
