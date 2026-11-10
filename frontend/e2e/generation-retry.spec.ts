import { expect, test } from "@playwright/test";

import { mockBackend, rememberAnalysedClip, variants } from "./support/backend";

test("a backend 500 mid-generation, then a successful retry", async ({ page }) => {
  let attempts = 0;
  await rememberAnalysedClip(page);
  await mockBackend(page, {
    generateVariants: (route) => {
      attempts += 1;
      return attempts === 1
        ? route.fulfill({
            status: 500,
            contentType: "application/json",
            body: JSON.stringify({ detail: "The model crashed." }),
          })
        : route.fulfill({ contentType: "application/json", body: JSON.stringify(variants) });
    },
  });
  await page.goto("/generate-variants");
  await page.getByRole("button", { name: /use my last upload/i }).click();
  const generate = page.getByRole("button", { name: /generate variants/i });

  await generate.click();
  await expect(page.getByText("The model crashed.")).toBeVisible();

  await generate.click();
  await expect(page.getByRole("heading", { name: /rendered variants/i })).toBeVisible();
  await expect(page.getByText("The model crashed.")).toBeHidden();
  expect(attempts).toBe(2);
});
