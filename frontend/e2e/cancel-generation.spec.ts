import { expect, test } from "@playwright/test";

import { mockBackend, rememberAnalysedClip } from "./support/backend";

test("cancelling mid-generation returns to idle with nothing left running", async ({ page }) => {
  await rememberAnalysedClip(page);
  let requested!: () => void;
  const requestArrived = new Promise<void>((resolve) => (requested = resolve));
  // Generation never answers on its own: only the user's cancel ends it.
  await mockBackend(page, {
    generateVariants: () => {
      requested();
      return new Promise<void>(() => {});
    },
  });
  await page.goto("/generate-variants");
  await page.getByRole("button", { name: /use my last upload/i }).click();
  const generate = page.getByRole("button", { name: /generate variants/i });

  const aborted = page.waitForEvent("requestfailed", (request) =>
    request.url().endsWith("/api/generate-variants"),
  );
  await generate.click();
  await requestArrived;
  await page.getByRole("button", { name: "Cancel" }).click();

  expect((await aborted).failure()?.errorText).toMatch(/abort/i);
  await expect(generate).toBeEnabled();
  await expect(generate).toBeFocused();
  await expect(page.getByRole("button", { name: "Cancel" })).toBeHidden();
  await expect(page.getByText(/configure on the left/i)).toBeVisible();
  await expect(page.getByRole("main").getByRole("alert")).toHaveCount(0);
});
