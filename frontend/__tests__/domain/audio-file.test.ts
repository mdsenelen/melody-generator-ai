import { checkAudioFile, MAX_UPLOAD_BYTES } from "../../src/domain/audioFile";

// Plain objects: the check only reads name, type and size, so the domain
// stays usable without a DOM File (and these tests run in the node project).
function file(name: string, type: string, size = 1024) {
  return { name, type, size };
}

describe("checkAudioFile", () => {
  it.each([
    ["take.wav", "audio/wav"],
    ["take.mp3", "audio/mpeg"],
    ["take.flac", "audio/flac"],
    ["take.ogg", "audio/ogg"],
    ["take.m4a", "audio/x-m4a"],
    ["take.webm", "audio/webm"],
    ["TAKE.WAV", "audio/wav"],
  ])("accepts %s (%s)", (name, type) => {
    expect(checkAudioFile(file(name, type))).toBeNull();
  });

  it("accepts a file the browser couldn't type, going by its extension", () => {
    expect(checkAudioFile(file("take.flac", ""))).toBeNull();
  });

  it.each([
    ["take.webm", "video/webm"],
    ["take.m4a", "video/mp4"],
    ["take.ogg", "video/ogg"],
  ])("accepts %s labelled %s, a container browsers often mislabel", (name, type) => {
    expect(checkAudioFile(file(name, type))).toBeNull();
  });

  it("rejects an extension the server can't decode", () => {
    expect(checkAudioFile(file("take.aiff", "audio/aiff"))).toEqual({
      reason: "unsupported_extension",
      message: "“take.aiff” isn't a supported audio format. Use WAV, MP3, FLAC, OGG, M4A or WebM.",
    });
  });

  it("rejects a file with no extension", () => {
    expect(checkAudioFile(file("take", "audio/wav"))?.reason).toBe("unsupported_extension");
  });

  it("rejects a non-audio file renamed to an audio extension", () => {
    expect(checkAudioFile(file("photo.wav", "image/png"))).toEqual({
      reason: "unsupported_type",
      message: "“photo.wav” isn't an audio file (it's image/png). Choose an audio recording.",
    });
  });

  it("rejects a zero-byte file", () => {
    expect(checkAudioFile(file("take.wav", "audio/wav", 0))).toEqual({
      reason: "empty",
      message: "“take.wav” is empty (0 bytes). Choose a file with audio in it.",
    });
  });

  it("rejects a file over the 50 MB limit", () => {
    expect(checkAudioFile(file("take.wav", "audio/wav", MAX_UPLOAD_BYTES + 1))).toEqual({
      reason: "too_large",
      message: "File is too large. The limit is 50 MB.",
    });
  });

  it("accepts a file of exactly 50 MB", () => {
    expect(checkAudioFile(file("take.wav", "audio/wav", MAX_UPLOAD_BYTES))).toBeNull();
  });

  it("reports an empty file as empty even when its type is also wrong", () => {
    expect(checkAudioFile(file("notes.txt", "text/plain", 0))?.reason).toBe("empty");
  });
});
