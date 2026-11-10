import { queryKeys } from "../../src/domain/queryKeys";

describe("queryKeys", () => {
  it("builds object-based keys, never concatenated strings", () => {
    expect(queryKeys.job("job-1")).toEqual([{ scope: "job", jobId: "job-1" }]);
    expect(queryKeys.progressions()).toEqual([{ scope: "progressions" }]);
    expect(queryKeys.clipAnalysis("job-1", 0, 60)).toEqual([
      { scope: "clipAnalysis", jobId: "job-1", startSec: 0, endSec: 60 },
    ]);
  });

  it("gives each clip window of a job its own key", () => {
    expect(queryKeys.clipAnalysis("job-1", 0, 60)).not.toEqual(
      queryKeys.clipAnalysis("job-1", 10, 60),
    );
    expect(queryKeys.clipAnalysis("job-1", 0, 60)).not.toEqual(
      queryKeys.clipAnalysis("job-2", 0, 60),
    );
  });

  it("keeps a job's key apart from its clip analyses", () => {
    const [job] = queryKeys.job("job-1");
    const [clip] = queryKeys.clipAnalysis("job-1", 0, 60);
    expect(job.scope).not.toBe(clip.scope);
  });

  it("has a disabled clip-analysis key that no real window can collide with", () => {
    expect(queryKeys.noClipAnalysis()).toEqual([{ scope: "clipAnalysis", jobId: null }]);
  });
});
