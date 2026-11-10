import type { AudioSource, Transcribing, TranscribeStep } from "../../src/domain/pipeline";
import {
  describeTranscription,
  describeVariantsProgress,
  WAKING_STATUS_MESSAGE,
} from "../../src/domain/statusCopy";

const source: AudioSource = { kind: "recording", file: new File(["x"], "hum.webm") };

function at(step: TranscribeStep): Transcribing {
  return { status: "transcribing", runId: 1, source, path: "server", upload: null, step };
}

describe("describeTranscription (/analyse)", () => {
  it.each<[TranscribeStep, string]>([
    [{ kind: "starting" }, "Transcribing hum.webm..."],
    [{ kind: "uploading", phase: "uploading" }, "Uploading hum.webm..."],
    [{ kind: "uploading", phase: "waking" }, WAKING_STATUS_MESSAGE],
    [
      { kind: "stillWorking" },
      "Still transcribing hum.webm... this can take a minute or two if the server was idle.",
    ],
    [{ kind: "browser", step: 0 }, "Transcribing hum.webm in your browser... 0%"],
    [{ kind: "browser", step: 7 }, "Transcribing hum.webm in your browser... 70%"],
    [{ kind: "browser", step: 10 }, "Transcribed hum.webm in your browser. Saving the notes..."],
    [{ kind: "savingRetry" }, "Server was still waking up — saving the notes for hum.webm..."],
    [{ kind: "submitRetry" }, "Server was still waking up — retrying hum.webm..."],
    [{ kind: "queued", slow: false }, "Queued for transcription..."],
    [
      { kind: "queued", slow: true },
      "Still queued for hum.webm... this can take a minute or two if the server was idle.",
    ],
    [{ kind: "processing", slow: false }, "Transcribing hum.webm..."],
    [
      { kind: "processing", slow: true },
      "Still transcribing hum.webm... longer clips take longer, and a cold server can add a minute or two on top.",
    ],
    [
      { kind: "warmingUp" },
      "The server is starting up. Your clip will be transcribed automatically, this can take about 2 to 4 minutes.",
    ],
    [
      { kind: "restarting" },
      "The server restarted and is coming back up. Your clip is safe -- we'll keep checking automatically.",
    ],
  ])("%j", (step, copy) => {
    expect(describeTranscription(at(step))).toBe(copy);
  });
});

describe("describeVariantsProgress (/generate-variants)", () => {
  it("shows browser progress, including 100%", () => {
    expect(describeVariantsProgress(at({ kind: "browser", step: 4 }))).toBe(
      "Transcribing hum.webm in your browser... 40%",
    );
    expect(describeVariantsProgress(at({ kind: "browser", step: 10 }))).toBe(
      "Transcribing hum.webm in your browser... 100%",
    );
  });

  it("says generating otherwise (unchanged from before the refactor)", () => {
    expect(describeVariantsProgress(at({ kind: "starting" }))).toBe("Generating variants");
    expect(describeVariantsProgress({ status: "generating", runId: 1, source })).toBe(
      "Generating variants",
    );
  });
});
