import { expect, test } from "@playwright/test";

import { mockBackend, TRANSCRIBE_JOB_ID, unreadableWav, type JobState } from "./support/backend";

test.describe("reloading during a running job", () => {
  test("the result page picks the job back up and shows it once done", async ({ page }) => {
    let job: JobState = "processing";
    await mockBackend(page, { transcribeJob: () => job });

    await page.goto(`/result/${TRANSCRIBE_JOB_ID}`);
    await expect(page.getByRole("status")).toBeVisible();

    await page.reload();
    await expect(page.getByRole("status")).toBeVisible();
    job = "completed";

    await expect(page.getByRole("link", { name: /download midi/i })).toBeVisible();
  });

  test("/analyse comes back idle and ready for a new upload", async ({ page }) => {
    await mockBackend(page, { transcribeJob: () => "processing" });
    await page.goto("/analyse");
    await page.getByLabel("Upload audio file").setInputFiles(unreadableWav("riff.wav"));
    await expect(page.getByRole("status")).toBeVisible();

    await page.reload();

    await expect(page.getByLabel("Upload audio file")).toBeEnabled();
    await expect(page.getByRole("status")).toHaveCount(0);
    await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
  });
});
