import {
  IDLE,
  sourceFile,
  sourceName,
  transition,
  type AudioSource,
  type PipelineEvent,
  type PipelineState,
} from "../../src/domain/pipeline";
import type { DomainError } from "../../src/domain/errors";

type Artifact = { id: string };
type State = PipelineState<Artifact>;

const file = new File(["x"], "take.wav", { type: "audio/wav" });
const upload: AudioSource = { kind: "upload", file, stored: { id: "u1", filename: "f.wav" } };
const failure: DomainError = { code: "failed", message: "boom", retryable: true };

function run(events: PipelineEvent<Artifact>[], from: State = IDLE): State {
  return events.reduce<State>((state, event) => transition(state, event), from);
}

function started(runId = 1): State {
  return run([{ type: "start", runId, source: upload }]);
}

function stepOf(state: State) {
  if (state.status !== "transcribing")
    throw new Error(`expected transcribing, got ${state.status}`);
  return state.step;
}

describe("sources", () => {
  it("names each kind of source", () => {
    expect(sourceName(upload)).toBe("take.wav");
    expect(sourceName({ kind: "recording", file })).toBe("take.wav");
    expect(sourceName({ kind: "session", sourceName: "earlier.mp3" })).toBe("earlier.mp3");
  });

  it("has a file only for uploads and recordings", () => {
    expect(sourceFile(upload)).toBe(file);
    expect(sourceFile({ kind: "session", sourceName: "earlier.mp3" })).toBeNull();
  });
});

describe("transition", () => {
  it("starts in the browser with no upload in flight", () => {
    expect(started()).toEqual({
      status: "transcribing",
      runId: 1,
      source: upload,
      path: "browser",
      upload: null,
      step: { kind: "starting" },
    });
  });

  it("a new start replaces any state, including a finished run", () => {
    const done = run([{ type: "completed", runId: 1, artifact: { id: "a" } }], started(1));
    const next = transition(done, { type: "start", runId: 2, source: upload });
    expect(next).toMatchObject({ status: "transcribing", runId: 2 });
  });

  describe("stale-run rule", () => {
    it("ignores every event from an earlier run", () => {
      const current = started(2);
      const stale: PipelineEvent<Artifact>[] = [
        { type: "browserProgress", runId: 1, fraction: 0.5 },
        { type: "failed", runId: 1, error: failure },
        { type: "completed", runId: 1, artifact: { id: "old" } },
        { type: "cancel", runId: 1 },
        { type: "submitted", runId: 1 },
      ];
      for (const event of stale) {
        expect(transition(current, event)).toBe(current);
      }
    });

    it("ignores run events while idle", () => {
      expect(transition(IDLE, { type: "browserProgress", runId: 1, fraction: 0.5 })).toBe(IDLE);
    });

    it("ignores progress after the run finished", () => {
      const done = run([{ type: "completed", runId: 1, artifact: { id: "a" } }], started());
      expect(transition(done, { type: "browserProgress", runId: 1, fraction: 0.9 })).toBe(done);
      expect(transition(done, { type: "failed", runId: 1, error: failure })).toBe(done);
    });
  });

  describe("browser progress", () => {
    it("rounds down to 10% steps and clamps at 10", () => {
      expect(
        stepOf(run([{ type: "browserProgress", runId: 1, fraction: 0.37 }], started())),
      ).toEqual({ kind: "browser", step: 3 });
      expect(
        stepOf(run([{ type: "browserProgress", runId: 1, fraction: 1.4 }], started())),
      ).toEqual({
        kind: "browser",
        step: 10,
      });
    });

    it("returns the same state for a repeat of the current step", () => {
      const at30 = run([{ type: "browserProgress", runId: 1, fraction: 0.31 }], started());
      expect(transition(at30, { type: "browserProgress", runId: 1, fraction: 0.38 })).toBe(at30);
    });
  });

  describe("upload phases", () => {
    it("are tracked but not shown while the browser path runs", () => {
      const state = run([{ type: "uploadPhase", runId: 1, phase: "waking" }], started());
      expect(state).toMatchObject({ upload: "waking", step: { kind: "starting" } });
    });

    it("are shown once the run fell back to the server", () => {
      const state = run(
        [
          { type: "fellBack", runId: 1 },
          { type: "uploadPhase", runId: 1, phase: "uploading" },
        ],
        started(),
      );
      expect(state).toMatchObject({ path: "server", upload: "uploading" });
      expect(stepOf(state)).toEqual({ kind: "uploading", phase: "uploading" });
    });

    it("falling back keeps the last status on screen", () => {
      const state = run(
        [
          { type: "browserProgress", runId: 1, fraction: 0.2 },
          { type: "fellBack", runId: 1 },
        ],
        started(),
      );
      expect(stepOf(state)).toEqual({ kind: "browser", step: 2 });
    });

    it("clears the phase once the upload finished", () => {
      const state = run(
        [
          { type: "uploadPhase", runId: 1, phase: "uploading" },
          { type: "uploaded", runId: 1 },
        ],
        started(),
      );
      expect(state).toMatchObject({ upload: null });
    });
  });

  describe("still-working timer", () => {
    it("is ignored on the browser path", () => {
      const state = started();
      expect(transition(state, { type: "stillWorking", runId: 1 })).toBe(state);
    });

    it("is ignored while the server is being woken", () => {
      const waking = run(
        [
          { type: "fellBack", runId: 1 },
          { type: "uploadPhase", runId: 1, phase: "waking" },
        ],
        started(),
      );
      expect(transition(waking, { type: "stillWorking", runId: 1 })).toBe(waking);
    });

    it("replaces the status on the server path", () => {
      const state = run(
        [
          { type: "fellBack", runId: 1 },
          { type: "uploadPhase", runId: 1, phase: "uploading" },
          { type: "stillWorking", runId: 1 },
        ],
        started(),
      );
      expect(stepOf(state)).toEqual({ kind: "stillWorking" });
    });
  });

  describe("server job", () => {
    const onServer = () => run([{ type: "fellBack", runId: 1 }], started());

    it("is queued once submitted", () => {
      expect(stepOf(run([{ type: "submitted", runId: 1 }], onServer()))).toEqual({
        kind: "queued",
        slow: false,
      });
    });

    it("maps job statuses, marking them slow after 8s", () => {
      const status = (status: string, elapsedMs: number, waitingReason?: "warming_up" | null) =>
        stepOf(
          run([{ type: "serverStatus", runId: 1, status, elapsedMs, waitingReason }], onServer()),
        );
      expect(status("queued", 1000)).toEqual({ kind: "queued", slow: false });
      expect(status("queued", 9000)).toEqual({ kind: "queued", slow: true });
      expect(status("processing", 8000)).toEqual({ kind: "processing", slow: false });
      expect(status("processing", 8001)).toEqual({ kind: "processing", slow: true });
      expect(status("queued", 1000, "warming_up")).toEqual({ kind: "warmingUp" });
    });

    it("leaves the status alone for other job statuses", () => {
      const state = run([{ type: "submitted", runId: 1 }], onServer());
      expect(
        transition(state, { type: "serverStatus", runId: 1, status: "completed", elapsedMs: 0 }),
      ).toBe(state);
    });

    it("shows retries and restarts", () => {
      expect(stepOf(run([{ type: "submitRetry", runId: 1 }], onServer()))).toEqual({
        kind: "submitRetry",
      });
      expect(stepOf(run([{ type: "restarting", runId: 1 }], onServer()))).toEqual({
        kind: "restarting",
      });
      expect(stepOf(run([{ type: "savingRetry", runId: 1 }], started()))).toEqual({
        kind: "savingRetry",
      });
    });
  });

  describe("finishing", () => {
    it("transcribing → generating → completed", () => {
      const generating = run([{ type: "generationStarted", runId: 1 }], started());
      expect(generating).toEqual({ status: "generating", runId: 1, source: upload });
      expect(
        transition(generating, { type: "completed", runId: 1, artifact: { id: "v" } }),
      ).toEqual({ status: "completed", runId: 1, source: upload, artifact: { id: "v" } });
    });

    it("transcribing → completed directly", () => {
      expect(
        run([{ type: "completed", runId: 1, artifact: { id: "t" } }], started()),
      ).toMatchObject({ status: "completed", artifact: { id: "t" } });
    });

    it("generating → failed, not yet dismissed", () => {
      const state = run(
        [
          { type: "generationStarted", runId: 1 },
          { type: "failed", runId: 1, error: failure },
        ],
        started(),
      );
      expect(state).toEqual({
        status: "failed",
        runId: 1,
        source: upload,
        error: failure,
        dismissed: false,
      });
    });

    it("dismissing a failure keeps the source", () => {
      const failed = run([{ type: "failed", runId: 1, error: failure }], started());
      expect(transition(failed, { type: "dismissError" })).toMatchObject({
        status: "failed",
        source: upload,
        dismissed: true,
      });
    });

    it("dismissing outside a failure changes nothing", () => {
      const state = started();
      expect(transition(state, { type: "dismissError" })).toBe(state);
    });

    it("cancel returns a running pipeline to idle", () => {
      expect(run([{ type: "cancel", runId: 1 }], started())).toBe(IDLE);
      const generating = run([{ type: "generationStarted", runId: 1 }], started());
      expect(transition(generating, { type: "cancel", runId: 1 })).toBe(IDLE);
    });
  });
});
