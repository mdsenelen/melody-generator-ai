import { useSessionStore } from "../../app/lib/session-store";

describe("useSessionStore", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    useSessionStore.setState({ lastUpload: null });
  });

  it("starts with no last upload", () => {
    expect(useSessionStore.getState().lastUpload).toBeNull();
  });

  it("stores an upload session and persists it to sessionStorage", () => {
    const session = {
      uploadId: "abc123",
      filename: "upload_abc123.wav",
      sourceName: "my-clip.wav",
      transcription: {
        chords: ["C", "G", "Am"],
        key: "C major",
        moodLabel: "happy" as const,
        pitchHistogram: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      },
    };

    useSessionStore.getState().setLastUpload(session);

    expect(useSessionStore.getState().lastUpload).toEqual(session);

    const raw = window.sessionStorage.getItem("melody-session");
    expect(raw).not.toBeNull();
    const parsed = JSON.parse(raw as string);
    expect(parsed.state.lastUpload).toEqual(session);
  });

  it("clears the last upload", () => {
    useSessionStore.getState().setLastUpload({
      uploadId: "x",
      filename: "upload_x.wav",
      sourceName: "clip.wav",
      transcription: { chords: [], key: "C major", moodLabel: "neutral", pitchHistogram: [] },
    });

    useSessionStore.getState().clearLastUpload();

    expect(useSessionStore.getState().lastUpload).toBeNull();
  });
});
