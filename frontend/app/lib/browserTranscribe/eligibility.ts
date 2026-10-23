// Whether a file should be transcribed in the browser or go straight to the
// server. Decided BEFORE decoding, because decoding is the memory spike:
// decodeAudioData holds the whole clip as 32-bit float PCM. Worst case is
// 48 kHz stereo: duration x 48000 x 2 ch x 4 B = 384 KB per second, so
//   300 s -> ~115 MB  (plus a ~26 MB 22.05 kHz mono copy for Basic Pitch)
//   600 s -> ~230 MB  (the server's MAX_UPLOAD_DURATION_SEC)
// 300 s keeps the spike well inside an ordinary tab. Devices that report
// little memory (navigator.deviceMemory, Chromium-only, rounded down to a
// power of two) get a tighter cap: 120 s -> ~46 MB.
export const BROWSER_MAX_DURATION_SEC = 300;
export const LOW_MEMORY_DEVICE_GB = 4;
export const LOW_MEMORY_MAX_DURATION_SEC = 120;
const PROBE_TIMEOUT_MS = 5000;

export type IneligibleReason =
  "too_long" | "low_memory_too_long" | "duration_unknown" | "unsupported";

export type Eligibility =
  { eligible: true; durationSec: number } | { eligible: false; reason: IneligibleReason };

export function browserPathEligibility({
  durationSec,
  deviceMemoryGb,
}: {
  durationSec: number | null;
  deviceMemoryGb?: number;
}): Eligibility {
  if (durationSec === null) {
    return { eligible: false, reason: "duration_unknown" };
  }
  if (durationSec > BROWSER_MAX_DURATION_SEC) {
    return { eligible: false, reason: "too_long" };
  }
  if (
    deviceMemoryGb !== undefined &&
    deviceMemoryGb < LOW_MEMORY_DEVICE_GB &&
    durationSec > LOW_MEMORY_MAX_DURATION_SEC
  ) {
    return { eligible: false, reason: "low_memory_too_long" };
  }
  return { eligible: true, durationSec };
}

// Reads the duration from the container header via an <audio> element --
// no full decode, so a 10-minute file costs nothing to reject. null when the
// browser can't read it (unsupported codec, or no metadata within the
// timeout); the server, which has ffmpeg, then handles it.
export function probeDurationSec(file: File, timeoutMs = PROBE_TIMEOUT_MS): Promise<number | null> {
  return new Promise((resolve) => {
    const audio = document.createElement("audio");
    const url = URL.createObjectURL(file);
    const finish = (value: number | null) => {
      window.clearTimeout(timer);
      audio.removeAttribute("src");
      audio.load();
      URL.revokeObjectURL(url);
      resolve(value);
    };
    const timer = window.setTimeout(() => finish(null), timeoutMs);
    audio.preload = "metadata";
    audio.onloadedmetadata = () =>
      finish(Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : null);
    audio.onerror = () => finish(null);
    audio.src = url;
  });
}

function supportsBrowserTranscription(): boolean {
  return typeof Worker !== "undefined" && typeof OfflineAudioContext !== "undefined";
}

export async function checkBrowserEligibility(file: File): Promise<Eligibility> {
  if (!supportsBrowserTranscription()) {
    return { eligible: false, reason: "unsupported" };
  }
  const deviceMemoryGb = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return browserPathEligibility({ durationSec: await probeDurationSec(file), deviceMemoryGb });
}
