// Split out of transcribeInBrowser.ts so tests can stub WebAudio decoding
// (jsdom has no OfflineAudioContext) while still running the real worker
// runner around it.

// Basic Pitch's required input rate (AUDIO_SAMPLE_RATE in @spotify/basic-pitch).
const BASIC_PITCH_SAMPLE_RATE = 22050;

export class BrowserDecodeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BrowserDecodeError";
  }
}

export type DecodedAudio = { audio: Float32Array; durationSec: number };

// decodeAudioData on an OfflineAudioContext resamples to that context's
// rate, so one native call both decodes and resamples to 22050 Hz -- no
// second full-length buffer at the source rate kept around in JS.
export async function decodeToMono22050(file: File): Promise<DecodedAudio> {
  let buffer: AudioBuffer;
  try {
    const context = new OfflineAudioContext(1, 1, BASIC_PITCH_SAMPLE_RATE);
    buffer = await context.decodeAudioData(await file.arrayBuffer());
  } catch (error) {
    throw new BrowserDecodeError(error instanceof Error ? error.message : "Could not decode audio");
  }
  const mono = new Float32Array(buffer.length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < data.length; i += 1) {
      mono[i] += data[i] / buffer.numberOfChannels;
    }
  }
  return { audio: mono, durationSec: buffer.duration };
}
