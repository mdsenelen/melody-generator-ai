import { writeFileSync } from "node:fs";

import { expect, test, type Page } from "@playwright/test";

import { mockBackend, unreadableWav } from "./support/backend";

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

function trackUploads(page: Page) {
  const sent: string[] = [];
  page.on("request", (request) => {
    if (/\/(upload|health)$/.test(new URL(request.url()).pathname)) sent.push(request.url());
  });
  return sent;
}

// Dropping a file skips the picker's type filter; setInputFiles does the same.
test.describe("a rejected file", () => {
  test("wrong format: says so, sends nothing, then accepts a real clip", async ({ page }) => {
    await mockBackend(page);
    const sent = trackUploads(page);
    await page.goto("/analyse");
    const input = page.getByLabel("Upload audio file");

    await input.setInputFiles({
      name: "notes.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("not audio"),
    });

    await expect(page.getByText(/isn't a supported audio format/i)).toBeVisible();
    expect(sent).toEqual([]);

    await input.setInputFiles(unreadableWav("riff.wav"));
    await expect(page.getByRole("heading", { name: "riff.wav" })).toBeVisible();
  });

  test("oversized: says so, sends nothing, then accepts a real clip", async ({
    page,
  }, testInfo) => {
    await mockBackend(page);
    const sent = trackUploads(page);
    await page.goto("/analyse");
    const input = page.getByLabel("Upload audio file");

    // Over Playwright's 50 MB in-memory limit, so it goes through a temp file.
    const oversized = testInfo.outputPath("long-take.wav");
    writeFileSync(oversized, Buffer.alloc(MAX_UPLOAD_BYTES + 1));
    await input.setInputFiles(oversized);

    await expect(page.getByText(/too large.*50 MB/i)).toBeVisible();
    expect(sent).toEqual([]);

    await input.setInputFiles(unreadableWav("riff.wav"));
    await expect(page.getByRole("heading", { name: "riff.wav" })).toBeVisible();
  });
});
