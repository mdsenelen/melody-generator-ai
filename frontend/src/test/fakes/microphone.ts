// jsdom has no MediaRecorder or getUserMedia. This fake records instantly:
// stop() emits one webm chunk, then onstop -- the same order a browser uses.

export class FakeMediaRecorder {
  static isTypeSupported() {
    return true;
  }

  state: "inactive" | "recording" = "inactive";
  mimeType = "audio/webm";
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;

  constructor(_stream: MediaStream) {
    void _stream;
  }

  start() {
    this.state = "recording";
  }

  stop() {
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["fake-audio"], { type: "audio/webm" }) });
    this.onstop?.();
  }
}

export function fakeStream(): MediaStream {
  return { getTracks: () => [] } as unknown as MediaStream;
}

/** A granted microphone: MediaRecorder + getUserMedia resolving a stream. */
export function installFakeMicrophone() {
  (globalThis as unknown as { MediaRecorder: unknown }).MediaRecorder = FakeMediaRecorder;
  Object.defineProperty(globalThis.navigator, "mediaDevices", {
    value: { getUserMedia: () => Promise.resolve(fakeStream()) },
    configurable: true,
  });
}
