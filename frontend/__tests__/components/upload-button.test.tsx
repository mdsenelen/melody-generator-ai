import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { UploadButton } from "../../components/upload-button";
import { uploadFile } from "../../app/lib/upload";
import { waitForBackend } from "../../app/lib/waitForBackend";

jest.mock("../../app/lib/upload", () => ({ uploadFile: jest.fn() }));
jest.mock("../../app/lib/waitForBackend", () => ({ waitForBackend: jest.fn() }));

const mockedUploadFile = uploadFile as jest.Mock;
const mockedWaitForBackend = waitForBackend as jest.Mock;

function networkError() {
  return Object.assign(new Error("Couldn't reach the server."), { code: "network_error" });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe("UploadButton", () => {
  beforeEach(() => {
    mockedWaitForBackend.mockResolvedValue(undefined);
    mockedUploadFile.mockResolvedValue({ id: "up-1", filename: "clip.wav" });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it("shows a caption under the drop zone only when given a label", () => {
    const { rerender } = render(<UploadButton onUploadSuccess={jest.fn()} />);
    expect(screen.queryByText(/^upload audio/i)).not.toBeInTheDocument();

    rerender(<UploadButton onUploadSuccess={jest.fn()} label="Upload audio" />);
    expect(screen.getByText("UPLOAD AUDIO")).toBeInTheDocument();
  });

  it("keeps only the animated uploading status visible while uploading", async () => {
    const pendingUpload = deferred<{ id: string; filename: string }>();
    mockedUploadFile.mockReturnValue(pendingUpload.promise);
    const user = userEvent.setup();
    render(<UploadButton onUploadSuccess={jest.fn()} />);

    await user.upload(screen.getByLabelText("Upload audio file"), new File(["audio"], "clip.wav"));

    expect(screen.queryByText("Uploading...")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Uploading");

    await act(async () => pendingUpload.resolve({ id: "up-1", filename: "clip.wav" }));
  });

  it("shows upload completion only briefly after a successful upload", async () => {
    const user = userEvent.setup();
    render(<UploadButton onUploadSuccess={jest.fn()} />);

    expect(screen.queryByText("Upload complete")).not.toBeInTheDocument();

    await user.upload(screen.getByLabelText("Upload audio file"), new File(["audio"], "clip.wav"));

    expect(screen.getByText("Upload complete")).toBeInTheDocument();

    await waitFor(() => expect(screen.queryByText("Upload complete")).not.toBeInTheDocument(), {
      timeout: 3000,
    });
  });

  describe("server waking up", () => {
    const originalFetch = global.fetch;

    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
      global.fetch = originalFetch;
    });

    it("shows a calm waking status while the server wakes, then uploads once it's up", async () => {
      const awake = deferred<void>();
      mockedWaitForBackend.mockImplementation((options: { onWaiting?: () => void }) => {
        options.onWaiting?.();
        return awake.promise;
      });
      const onUploadSuccess = jest.fn();
      const onUploadError = jest.fn();
      render(<UploadButton onUploadSuccess={onUploadSuccess} onUploadError={onUploadError} />);

      fireEvent.change(screen.getByLabelText("Upload audio file"), {
        target: { files: [new File(["audio"], "clip.wav")] },
      });

      await waitFor(() =>
        expect(screen.getByRole("status", { name: /waking the server/i })).toBeInTheDocument(),
      );
      expect(mockedUploadFile).not.toHaveBeenCalled();
      expect(onUploadError).not.toHaveBeenCalled();

      await act(async () => awake.resolve());

      await waitFor(() =>
        expect(onUploadSuccess).toHaveBeenCalledWith(
          // pickedAt: the moment the file was chosen, so /analyse can time
          // pick -> "Transcription ready" including this upload.
          expect.objectContaining({
            id: "up-1",
            filename: "clip.wav",
            pickedAt: expect.any(Number),
          }),
        ),
      );
      expect(mockedUploadFile).toHaveBeenCalledTimes(1);
      expect(onUploadError).not.toHaveBeenCalled();
    });

    it("never aborts or re-sends an upload that takes longer than 15s to send", async () => {
      let uploadSignal: AbortSignal | undefined;
      mockedUploadFile.mockImplementation(
        (_file: File, options: { signal?: AbortSignal }) =>
          new Promise((resolve) => {
            uploadSignal = options.signal;
            setTimeout(() => resolve({ id: "up-1", filename: "clip.wav" }), 40_000);
          }),
      );
      const onUploadSuccess = jest.fn();
      render(<UploadButton onUploadSuccess={onUploadSuccess} />);

      fireEvent.change(screen.getByLabelText("Upload audio file"), {
        target: { files: [new File(["audio"], "clip.wav")] },
      });

      await act(() => jest.advanceTimersByTimeAsync(40_000));

      await waitFor(() => expect(onUploadSuccess).toHaveBeenCalled());
      expect(mockedUploadFile).toHaveBeenCalledTimes(1);
      expect(uploadSignal?.aborted).not.toBe(true);
    });

    it("says the server is restarting or unreachable once it never wakes", async () => {
      mockedWaitForBackend.mockRejectedValue(networkError());
      // The reachability probe can't connect either.
      global.fetch = jest.fn().mockRejectedValue(new TypeError("Failed to fetch"));
      const onUploadError = jest.fn();
      render(<UploadButton onUploadSuccess={jest.fn()} onUploadError={onUploadError} />);

      fireEvent.change(screen.getByLabelText("Upload audio file"), {
        target: { files: [new File(["audio"], "clip.wav")] },
      });

      await waitFor(() =>
        expect(onUploadError).toHaveBeenCalledWith(
          "The server is restarting or unreachable — try again in a minute.",
        ),
      );
      expect(mockedUploadFile).not.toHaveBeenCalled();
    });
  });
});
