import { formatTranscribeLog } from "../../app/lib/transcribeTiming";

describe("formatTranscribeLog", () => {
  it("names the path, the TF.js backend, the note count and pick-to-ready seconds", () => {
    expect(
      formatTranscribeLog({ path: "browser", backend: "webgl", notes: 812, elapsedMs: 24_312 }),
    ).toBe("[transcribe] path=browser backend=webgl notes=812 elapsed=24.3s");
  });

  it("reports backend=none on the server path, where no TF.js backend ran", () => {
    expect(
      formatTranscribeLog({ path: "server", backend: null, notes: 790, elapsedMs: 271_049 }),
    ).toBe("[transcribe] path=server backend=none notes=790 elapsed=271.0s");
  });
});
