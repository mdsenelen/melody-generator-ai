import { MAX_UPLOAD_BYTES, uploadFile } from "../../app/lib/upload";

describe("uploadFile", () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("rejects an oversized file before sending anything", async () => {
    const fetchSpy = jest.fn();
    global.fetch = fetchSpy;
    const file = new File(["x"], "big.wav", { type: "audio/wav" });
    Object.defineProperty(file, "size", { value: MAX_UPLOAD_BYTES + 1 });

    await expect(uploadFile(file)).rejects.toThrow(/too large/i);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("replaces the opaque 'Failed to fetch' with an actionable message", async () => {
    global.fetch = jest.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    const file = new File(["x"], "a.wav", { type: "audio/wav" });

    await expect(uploadFile(file)).rejects.toThrow(/couldn't reach the server/i);
  });
});
