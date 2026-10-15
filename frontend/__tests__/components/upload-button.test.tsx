import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { UploadButton } from "../../components/upload-button";
import { uploadFile } from "../../app/lib/upload";
import {
  RETRY_ATTEMPT_TIMEOUT_MS,
  RETRY_INITIAL_DELAY_MS,
  RETRY_MAX_TOTAL_MS,
} from "../../app/lib/retryOnWakeup";

jest.mock("../../app/lib/upload", () => ({ uploadFile: jest.fn() }));

const mockedUploadFile = uploadFile as jest.Mock;

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
    mockedUploadFile.mockResolvedValue({ id: "up-1", filename: "clip.wav" });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it("keeps only the animated uploading status visible while uploading", async () => {
    const pendingUpload = deferred<{ id: string; filename: string }>();
    mockedUploadFile.mockReturnValue(pendingUpload.promise);
    const user = userEvent.setup();
    render(<UploadButton onUploadSuccess={jest.fn()} />);

    await user.upload(screen.getByLabelText("Upload audio file"), new File(["audio"], "clip.wav"));

    expect(screen.queryByText("Uploading...")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Uploading");

    pendingUpload.resolve({ id: "up-1", filename: "clip.wav" });
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
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it("shows a calm waking-up status instead of the raw error, then succeeds once the retry lands", async () => {
      mockedUploadFile
        .mockRejectedValueOnce(networkError())
        .mockResolvedValueOnce({ id: "up-1", filename: "clip.wav" });
      const onUploadSuccess = jest.fn();
      const onUploadError = jest.fn();
      render(<UploadButton onUploadSuccess={onUploadSuccess} onUploadError={onUploadError} />);

      fireEvent.change(screen.getByLabelText("Upload audio file"), {
        target: { files: [new File(["audio"], "clip.wav")] },
      });

      await waitFor(() => expect(screen.getByText(/waking up/i)).toBeInTheDocument());
      expect(onUploadError).not.toHaveBeenCalled();

      await act(() => jest.advanceTimersByTimeAsync(RETRY_INITIAL_DELAY_MS));

      await waitFor(() =>
        expect(onUploadSuccess).toHaveBeenCalledWith(
          expect.objectContaining({ id: "up-1", filename: "clip.wav" }),
        ),
      );
      expect(onUploadError).not.toHaveBeenCalled();
    });

    it("falls back to the real error message only once the retry budget is exhausted", async () => {
      mockedUploadFile.mockRejectedValue(networkError());
      const onUploadError = jest.fn();
      render(<UploadButton onUploadSuccess={jest.fn()} onUploadError={onUploadError} />);

      fireEvent.change(screen.getByLabelText("Upload audio file"), {
        target: { files: [new File(["audio"], "clip.wav")] },
      });

      expect(onUploadError).not.toHaveBeenCalled();

      await act(() => jest.advanceTimersByTimeAsync(RETRY_MAX_TOTAL_MS + RETRY_ATTEMPT_TIMEOUT_MS));

      await waitFor(() => expect(onUploadError).toHaveBeenCalledWith("Couldn't reach the server."));
    });
  });
});
