import { readFileSync } from "node:fs";

import { expect, test } from "@playwright/test";

import { mockBackend, unreadableWav, variants } from "./support/backend";

test("upload, analyse, generate, play back and export MIDI", async ({ page }) => {
  await mockBackend(page);

  await test.step("upload and analyse", async () => {
    await page.goto("/analyse");
    await page.getByLabel("Upload audio file").setInputFiles(unreadableWav("riff.wav"));
    await expect(page.getByRole("heading", { name: "riff.wav" })).toBeVisible();
    await expect(page.getByText("C major")).toBeVisible();
  });

  await test.step("generate variants from the analysed clip", async () => {
    await page
      .getByRole("link", { name: /variants/i })
      .first()
      .click();
    await page.getByRole("button", { name: /use my last upload/i }).click();
    await page.getByRole("button", { name: /generate variants/i }).click();
    await expect(page.getByRole("heading", { name: /rendered variants/i })).toBeVisible();
  });

  await test.step("play the melody", async () => {
    await page.getByRole("button", { name: "Play melody" }).click();
    await expect(page.getByRole("button", { name: "Pause melody" })).toBeVisible();
    await page.getByRole("button", { name: "Pause melody" }).click();
  });

  await test.step("export the MIDI", async () => {
    await page.getByRole("link", { name: /view & download result/i }).click();
    const downloading = page.waitForEvent("download");
    await page.getByRole("link", { name: /download midi/i }).click();
    const download = await downloading;

    expect(download.suggestedFilename()).toBe("variant_1.mid");
    const bytes = readFileSync(await download.path());
    expect(bytes.subarray(0, 4).toString("latin1")).toBe("MThd");
    expect(bytes.equals(Buffer.from(variants.variants[0].midi_b64, "base64"))).toBe(true);
  });
});
