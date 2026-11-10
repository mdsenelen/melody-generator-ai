import { defineConfig, devices } from "@playwright/test";

// The suite runs against `next dev` with every FastAPI route intercepted in
// the browser (e2e/support/backend.ts), so no test ever reaches a model.
// Its own port, so it never collides with (or reuses) a dev server you
// already have open on :3000.
const PORT = 3100;
// Stub for the server-side calls the browser can't intercept (downloads).
const STUB_BACKEND_PORT = 8100;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  timeout: 30_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: `node e2e/support/stub-backend.mjs ${STUB_BACKEND_PORT}`,
      url: `http://127.0.0.1:${STUB_BACKEND_PORT}/health`,
      reuseExistingServer: false,
    },
    {
      command: `npm run dev -- --port ${PORT}`,
      url: `http://localhost:${PORT}`,
      reuseExistingServer: false,
      timeout: 120_000,
      env: {
        // The origin the browser talks to; e2e/support/backend.ts intercepts
        // it in the page, whatever .env.local says.
        NEXT_PUBLIC_BACKEND_URL: "http://127.0.0.1:8000",
        // Where the Next.js proxy routes fetch from, server-side.
        BACKEND_BASE_URL: `http://127.0.0.1:${STUB_BACKEND_PORT}`,
      },
    },
  ],
});
