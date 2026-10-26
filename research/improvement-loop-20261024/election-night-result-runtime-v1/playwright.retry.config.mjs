import { defineConfig, devices } from "/home/hayden/projects/ballotmancy-election-night-guide-20261026/node_modules/@playwright/test/index.mjs";
export default defineConfig({
  testDir: "/home/hayden/projects/ballotmancy-election-night-guide-20261026/tests/site",
  testMatch: "election-night.spec.ts", fullyParallel: false, workers: 1, timeout: 45000, retries: 0,
  outputDir: "/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/election-night-result-runtime-v1/browser-retry-artifacts",
  reporter: [["list"], ["json", { outputFile: "/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/election-night-result-runtime-v1/browser-retry-results.json" }]],
  use: { baseURL: "http://127.0.0.1:3137", trace: "retain-on-failure", screenshot: "only-on-failure" },
  projects: [
    { name: "desktop-chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile-chromium", use: { ...devices["Pixel 7"] } }
  ]
});
