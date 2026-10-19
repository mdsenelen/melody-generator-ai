import { explainUploadNetworkError } from "../../app/lib/explainUploadNetworkError";
import { RetryCancelledError } from "../../app/lib/retryOnWakeup";
import { uploadFile } from "../../app/lib/upload";
import { uploadWhenAwake } from "../../app/lib/uploadWhenAwake";
import { waitForBackend } from "../../app/lib/waitForBackend";

jest.mock("../../app/lib/upload", () => ({ uploadFile: jest.fn() }));
jest.mock("../../app/lib/waitForBackend", () => ({ waitForBackend: jest.fn() }));
jest.mock("../../app/lib/explainUploadNetworkError", () => ({
  explainUploadNetworkError: jest.fn(),
}));

const mockedUploadFile = uploadFile as jest.Mock;
const mockedWaitForBackend = waitForBackend as jest.Mock;
const mockedExplain = explainUploadNetworkError as jest.Mock;

function networkError() {
  return Object.assign(new Error("Couldn't reach the server."), { code: "network_error" });
}

function httpError(status: number) {
  return Object.assign(new Error(`Request failed (${status})`), { status });
}

const file = new File(["audio"], "clip.wav", { type: "audio/wav" });

describe("uploadWhenAwake", () => {
  beforeEach(() => {
    jest.useFakeTimers();
    mockedWaitForBackend.mockResolvedValue(undefined);
    mockedExplain.mockImplementation(async (error: unknown) => error);
    mockedUploadFile.mockResolvedValue({ id: "up-1", filename: "upload_up-1.wav" });
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.clearAllMocks();
  });

  it("waits for the server before sending the upload, then sends it once", async () => {
    const order: string[] = [];
    mockedWaitForBackend.mockImplementation(async () => {
      order.push("wait");
    });
    mockedUploadFile.mockImplementation(async () => {
      order.push("upload");
      return { id: "up-1", filename: "upload_up-1.wav" };
    });

    await expect(uploadWhenAwake(file)).resolves.toEqual({
      id: "up-1",
      filename: "upload_up-1.wav",
    });
    expect(order).toEqual(["wait", "upload"]);
  });

  it("never aborts or re-sends a slow upload body (>15s)", async () => {
    let uploadSignal: AbortSignal | undefined;
    mockedUploadFile.mockImplementation(
      (_file: File, options: { signal?: AbortSignal }) =>
        new Promise((resolve) => {
          uploadSignal = options.signal;
          setTimeout(() => resolve({ id: "up-1", filename: "upload_up-1.wav" }), 45_000);
        }),
    );

    const promise = uploadWhenAwake(file);
    await jest.advanceTimersByTimeAsync(45_000);

    await expect(promise).resolves.toMatchObject({ id: "up-1" });
    expect(mockedUploadFile).toHaveBeenCalledTimes(1);
    expect(uploadSignal?.aborted).not.toBe(true);
  });

  it("reports waking while the server is waking, then uploading", async () => {
    mockedWaitForBackend.mockImplementation(async (options: { onWaiting?: () => void }) => {
      options.onWaiting?.();
    });
    const onPhase = jest.fn();

    await uploadWhenAwake(file, { onPhase });

    expect(onPhase.mock.calls.map(([phase]) => phase)).toEqual(["waking", "uploading"]);
  });

  it.each([
    ["a network error", networkError()],
    ["a bare 503", httpError(503)],
  ])("does one more wake-then-upload cycle after %s", async (_label, error) => {
    mockedUploadFile
      .mockRejectedValueOnce(error)
      .mockResolvedValueOnce({ id: "up-2", filename: "upload_up-2.wav" });

    await expect(uploadWhenAwake(file)).resolves.toMatchObject({ id: "up-2" });
    expect(mockedWaitForBackend).toHaveBeenCalledTimes(2);
    expect(mockedUploadFile).toHaveBeenCalledTimes(2);
  });

  it("gives up after the second failed cycle and rethrows", async () => {
    mockedUploadFile.mockRejectedValue(networkError());

    await expect(uploadWhenAwake(file)).rejects.toMatchObject({ code: "network_error" });
    expect(mockedUploadFile).toHaveBeenCalledTimes(2);
  });

  it("explains the final error, so the user can tell 'down' apart from 'rejected'", async () => {
    const original = networkError();
    const explained = Object.assign(new Error("explained"), { code: "network_error" });
    mockedUploadFile.mockRejectedValue(original);
    mockedExplain.mockResolvedValue(explained);
    const controller = new AbortController();

    await expect(uploadWhenAwake(file, { signal: controller.signal })).rejects.toBe(explained);
    expect(mockedExplain).toHaveBeenCalledWith(original, controller.signal);
  });

  it("explains a wake that never succeeds too", async () => {
    const explained = Object.assign(new Error("explained"), { code: "network_error" });
    mockedWaitForBackend.mockRejectedValue(networkError());
    mockedExplain.mockResolvedValue(explained);

    await expect(uploadWhenAwake(file)).rejects.toBe(explained);
    expect(mockedUploadFile).not.toHaveBeenCalled();
  });

  it("does not explain a cancellation", async () => {
    mockedWaitForBackend.mockRejectedValue(new RetryCancelledError());

    await expect(uploadWhenAwake(file)).rejects.toBeInstanceOf(RetryCancelledError);
    expect(mockedExplain).not.toHaveBeenCalled();
  });

  it("rethrows a non-wake error (e.g. a 400) immediately without re-sending", async () => {
    mockedUploadFile.mockRejectedValue(httpError(400));

    await expect(uploadWhenAwake(file)).rejects.toMatchObject({ status: 400 });
    expect(mockedUploadFile).toHaveBeenCalledTimes(1);
  });

  it("rejects with RetryCancelledError when the caller aborts mid-upload", async () => {
    mockedUploadFile.mockImplementation(
      (_file: File, options: { signal?: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          options.signal?.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          );
        }),
    );
    const controller = new AbortController();

    const promise = uploadWhenAwake(file, { signal: controller.signal });
    const assertion = expect(promise).rejects.toBeInstanceOf(RetryCancelledError);
    await jest.advanceTimersByTimeAsync(1_000);
    controller.abort();

    await assertion;
    expect(mockedUploadFile).toHaveBeenCalledTimes(1);
  });
});
