import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/site",
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  retries: process.env.CI ? 1 : 0,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:3127",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "desktop-chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile-chromium", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command: "npm run start -- --port 3127",
    url: "http://127.0.0.1:3127/api/health",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
