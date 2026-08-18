import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { AudioRecorder } from "../../components/audio-recorder";

class FakeMediaRecorder {
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

function fakeStream(): MediaStream {
  return { getTracks: () => [] } as unknown as MediaStream;
}

beforeEach(() => {
  (global as unknown as { MediaRecorder: unknown }).MediaRecorder = FakeMediaRecorder;
  Object.defineProperty(global.navigator, "mediaDevices", {
    value: { getUserMedia: jest.fn().mockResolvedValue(fakeStream()) },
    configurable: true,
  });
  global.URL.createObjectURL = jest.fn().mockReturnValue("blob:fake-url");
  global.URL.revokeObjectURL = jest.fn();
});

describe("AudioRecorder", () => {
  it("starts in a ready-to-record state with Stop disabled", () => {
    render(<AudioRecorder onRecordingComplete={jest.fn()} />);
    expect(screen.getByText("Ready to record")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /stop/i })).toBeDisabled();
  });

  it("goes through record -> stop -> preview -> use recording", async () => {
    const user = userEvent.setup();
    const onRecordingComplete = jest.fn();
    render(<AudioRecorder onRecordingComplete={onRecordingComplete} />);

    await user.click(screen.getByRole("button", { name: /record/i }));
    await waitFor(() => expect(screen.getByText("Recording...")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /record/i })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: /stop/i }));
    expect(await screen.findByRole("button", { name: /use this recording/i })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /use this recording/i }));

    expect(onRecordingComplete).toHaveBeenCalledTimes(1);
    const [file] = onRecordingComplete.mock.calls[0];
    expect(file).toBeInstanceOf(File);
  });

  it("lets the user discard a preview and record again", async () => {
    const user = userEvent.setup();
    render(<AudioRecorder onRecordingComplete={jest.fn()} />);

    await user.click(screen.getByRole("button", { name: /record/i }));
    await user.click(screen.getByRole("button", { name: /stop/i }));
    await screen.findByRole("button", { name: /use this recording/i });

    await user.click(screen.getByRole("button", { name: /record again/i }));

    expect(screen.getByText("Ready to record")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /use this recording/i })).not.toBeInTheDocument();
  });

  it("shows an error message when microphone access is denied", async () => {
    const user = userEvent.setup();
    Object.defineProperty(global.navigator, "mediaDevices", {
      value: {
        getUserMedia: jest.fn().mockRejectedValue(new Error("Permission denied")),
      },
      configurable: true,
    });

    render(<AudioRecorder onRecordingComplete={jest.fn()} />);
    await user.click(screen.getByRole("button", { name: /record/i }));

    expect(await screen.findByText("Permission denied")).toBeInTheDocument();
    expect(screen.getByText("Recording unavailable")).toBeInTheDocument();
  });
});
