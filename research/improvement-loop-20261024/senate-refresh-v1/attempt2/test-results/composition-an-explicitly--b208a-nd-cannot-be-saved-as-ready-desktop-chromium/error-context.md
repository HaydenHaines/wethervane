# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: composition.spec.ts >> an explicitly uncomputed fixture stays separate from the saved result and cannot be saved as ready
- Location: tests/site/composition.spec.ts:74:5

# Error details

```
Error: expect(locator).toContainText(expected) failed

Locator: getByRole('region', { name: 'Who could control the chamber?', exact: true })
Expected substring: "8.3%"
Received string:    "Who could control the chamber?Independents unaligned · D needs 51; R needs 50Democratic control43.5%Republican control53.2%Control depends on alliances3.3%King and Sanders continue working with the Democrats. The four modeled independent challengers are assigned to neither party if elected, consistent with their stated pledges. Their eventual decisions could differ. Unresolved control means the winners are known in a simulation, but the winning members’ party alliances have not determined who organizes the chamber.A caucus is the group working with a party to organize the chamber, including any allied independents. The party holding the vice presidency can control a 50–50 Senate through its tie-breaking vote.D caucus: 50.1 expected seats (46–55, modeled 90% range)R caucus: 49.6 expected seats (44–54, modeled 90% range)Unaligned: 0.3 expected seats (0–2, modeled 90% range)Expected seats are the average across simulated elections, so they can include fractions. The 90% range spans the 5th to 95th percentiles; outcomes outside that range remain possible."
Timeout: 5000ms

Call log:
  - Expect "toContainText" getByRole('region', { name: 'Who could control the chamber?', exact: true }) with timeout 5000ms
  - waiting for getByRole('region', { name: 'Who could control the chamber?', exact: true })
    12 × locator resolved to <section class="control-card" aria-labelledby="control-title">…</section>
       - unexpected value "Who could control the chamber?Independents unaligned · D needs 51; R needs 50Democratic control43.5%Republican control53.2%Control depends on alliances3.3%King and Sanders continue working with the Democrats. The four modeled independent challengers are assigned to neither party if elected, consistent with their stated pledges. Their eventual decisions could differ. Unresolved control means the winners are known in a simulation, but the winning members’ party alliances have not determined who organizes the chamber.A caucus is the group working with a party to organize the chamber, including any allied independents. The party holding the vice presidency can control a 50–50 Senate through its tie-breaking vote.D caucus: 50.1 expected seats (46–55, modeled 90% range)R caucus: 49.6 expected seats (44–54, modeled 90% range)Unaligned: 0.3 expected seats (0–2, modeled 90% range)Expected seats are the average across simulated elections, so they can include fractions. The 90% range spans the 5th to 95th percentiles; outcomes outside that range remain possible."

```

```yaml
- region "Who could control the chamber?":
  - heading "Who could control the chamber?" [level=2]
  - text: Independents unaligned · D needs 51; R needs 50 Democratic control 43.5% Republican control 53.2% Control depends on alliances 3.3%
  - paragraph: King and Sanders continue working with the Democrats. The four modeled independent challengers are assigned to neither party if elected, consistent with their stated pledges. Their eventual decisions could differ. Unresolved control means the winners are known in a simulation, but the winning members’ party alliances have not determined who organizes the chamber.
  - paragraph: A caucus is the group working with a party to organize the chamber, including any allied independents. The party holding the vice presidency can control a 50–50 Senate through its tie-breaking vote.
  - text: "D caucus: 50.1 expected seats (46–55, modeled 90% range) R caucus: 49.6 expected seats (44–54, modeled 90% range) Unaligned: 0.3 expected seats (0–2, modeled 90% range)"
  - paragraph: Expected seats are the average across simulated elections, so they can include fractions. The 90% range spans the 5th to 95th percentiles; outcomes outside that range remain possible.
```

# Test source

```ts
  1   | import { expect, test, type Page } from "@playwright/test";
  2   | import type { CompositionCatalog } from "../../lib/composition-types";
  3   | 
  4   | async function expandControls(page: Page) {
  5   |   const controls = page.locator("details.stack-builder");
  6   |   await expect(controls).toBeVisible();
  7   |   if (await controls.getAttribute("open") === null) await controls.locator(":scope > summary").click();
  8   |   await expect(controls).toHaveAttribute("open", "");
  9   | }
  10  | 
  11  | test("saved current research comparisons update the forecast and compare only matching definitions", async ({ page }) => {
  12  |   await page.goto("/?run=current-polling");
  13  |   await expandControls(page);
  14  |   await page.getByLabel("Base model", { exact: true }).selectOption("senate_relative_context_v1");
  15  |   await expect(page).toHaveURL(/run=current-comparison-senate_relative_context_v1-fp/);
  16  |   await expect(page.locator(".composition-state")).toContainText("Ready");
  17  |   await expect(page.locator(".stack-builder")).not.toContainText("No current forecast has been generated");
  18  |   await expect(page.getByRole("region", { name: "The path to a majority", exact: true })).toBeVisible();
  19  |   await expect(page.getByRole("region", { name: "Who could control the chamber?", exact: true })).toBeVisible();
  20  |   await page.getByText("Input and publication times", { exact: true }).click();
  21  |   await expect(page.locator(".edition-inputs")).toContainText("15:21:23 GMT");
  22  |   await expect(page.locator(".edition-inputs")).toContainText("16:21:44 GMT");
  23  |   await page.getByRole("checkbox", { name: "Compare default", exact: true }).check();
  24  |   await expect(page.locator(".compare-banner").filter({ hasText: "Comparing with" })).toContainText("different forecast definitions");
  25  |   await page.getByLabel("Comparison baseline", { exact: true }).selectOption("current-comparison-senate_relative_context_v1-f");
  26  |   await expect(page.locator(".compare-banner").filter({ hasText: "Comparing with" })).toContainText("Changes compare the same races");
  27  |   await page.getByRole("checkbox", { name: "Include polling evidence", exact: true }).uncheck();
  28  |   await expect(page).toHaveURL(/run=current-comparison-senate_relative_context_v1-f(?:&|$)/);
  29  |   await expect(page.getByRole("region", { name: "How this model did", exact: true })).toHaveCount(0);
  30  |   expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  31  | });
  32  | 
  33  | test("the current saved forecast starts with component controls collapsed", async ({ page }) => {
  34  |   await page.goto("/");
  35  |   await expect(page).toHaveURL(/run=current-polling/);
  36  |   const controls = page.locator("details.stack-builder");
  37  |   await expect(controls).toBeVisible();
  38  |   await expect(controls).not.toHaveAttribute("open", "");
  39  |   await expect(controls.locator(":scope > summary")).toContainText("Build your perspective");
  40  |   await expect(page.getByLabel("Base model", { exact: true })).not.toBeVisible();
  41  |   await expect(page.getByRole("region", { name: "The path to a majority", exact: true })).toBeVisible();
  42  |   await expect(page.getByRole("region", { name: "How this model did", exact: true })).toHaveCount(0);
  43  |   await expandControls(page);
  44  |   await expect(page.getByRole("checkbox", { name: "Include polling evidence", exact: true })).toBeChecked();
  45  |   await expect(page.getByLabel("Polling component", { exact: true })).toHaveCount(0);
  46  |   await expect(page.locator(".composition-state")).toContainText("Ready");
  47  |   await controls.locator(":scope > summary").click();
  48  |   await expect(controls).not.toHaveAttribute("open", "");
  49  |   await expect(page).toHaveURL(/run=current-polling/);
  50  | });
  51  | 
  52  | test("historical scorecard distinguishes outcome hits, probability error, and actual chamber control", async ({ page }) => {
  53  |   await page.goto("/?run=history-2018-FP-1");
  54  |   const scoreboard = page.getByRole("region", { name: "How this model did", exact: true });
  55  |   const favorites = scoreboard.getByRole("article", { name: "Favorites that won", exact: true });
  56  |   await expect(favorites.locator(".scoreboard-value")).toHaveText(/31\s*\/\s*35/);
  57  |   await expect(favorites).toContainText("31.2");
  58  |   await favorites.getByText("Explore 4 misses", { exact: true }).click();
  59  |   await favorites.getByRole("button", { name: "Maine", exact: true }).click();
  60  |   await expect(page.getByRole("complementary").getByRole("heading", { name: "Maine", exact: true })).toBeVisible();
  61  |   const quality = scoreboard.getByRole("article", { name: "Probability quality", exact: true });
  62  |   await quality.getByText("Probability score details", { exact: true }).click();
  63  |   await expect(quality).toContainText("Average Brier score across races: 0.1352");
  64  |   await expect(quality).toContainText("0–2 scale");
  65  |   await expect(quality).toContainText("35 contests scored.");
  66  |   await expect(scoreboard.getByRole("img", { name: /^D caucus: forecast median .*actual 47\./ })).toBeVisible();
  67  |   await expect(scoreboard.getByRole("img", { name: /^R caucus: forecast median .*actual 53\./ })).toBeVisible();
  68  |   const control = scoreboard.getByRole("article", { name: "Chamber control", exact: true });
  69  |   await expect(control.locator(".observed")).toContainText("R control · Actual");
  70  |   await expect(control.locator(".scoreboard-value")).toHaveText("99.1%");
  71  |   expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  72  | });
  73  | 
  74  | test("an explicitly uncomputed fixture stays separate from the saved result and cannot be saved as ready", async ({ page, request }) => {
  75  |   const catalog = await (await request.get("/api/configurations/catalog")).json() as CompositionCatalog;
  76  |   const experimental = catalog.computation.find((entry) => entry.years.includes(2026))!;
  77  |   // All currently supported 2026 combinations are now saved. Exercise future
  78  |   // pending-result behavior explicitly, without mislabeling a live configuration.
  79  |   await page.route("**/api/configurations/resolve", async (route) => {
  80  |     const selection = route.request().postDataJSON();
  81  |     if (selection.components.base !== experimental.baseId) return route.continue();
  82  |     await route.fulfill({ json: {
  83  |       state: "requires_computation", request: selection, configurationId: null,
  84  |       resultId: null, configuration: null, run: null,
  85  |       reason: "Synthetic uncomputed configuration for browser lifecycle testing.",
  86  |       requiredSteps: ["Fit the selected model jointly."], evidence: experimental.evidence,
  87  |       outputs: ["race-probabilities"], executionAvailable: false,
  88  |     } });
  89  |   });
  90  |   await page.goto("/?run=current-polling&race=2026%3ANE%3Asenate%3Aregular");
  91  |   const chamber = page.getByRole("region", { name: "Who could control the chamber?", exact: true });
> 92  |   await expect(chamber).toContainText("8.3%");
      |                         ^ Error: expect(locator).toContainText(expected) failed
  93  |   const savedChamber = await chamber.innerText();
  94  |   await expandControls(page);
  95  |   await page.getByLabel("Base model", { exact: true }).selectOption(experimental.baseId);
  96  |   const state = page.locator(".composition-state");
  97  |   await expect(state).toContainText("Requires computation");
  98  |   await expect(state).toContainText("Still displaying");
  99  |   await expect(page).toHaveURL(/run=current-polling/);
  100 |   await expect(chamber).toHaveText(savedChamber, {useInnerText:true});
  101 |   await expect(page.getByRole("complementary").getByText("14.2%", { exact: true })).toBeVisible();
  102 |   await expect(page.locator(".stack-builder-summary")).toContainText("Requires computation");
  103 |   await page.locator("details.composition-saved > summary").click();
  104 |   await page.getByLabel("Name this view", { exact: true }).fill("Uncomputed request");
  105 |   await expect(page.getByRole("button", { name: "Save view", exact: true })).toBeDisabled();
  106 |   await page.getByRole("button", { name: "Reset stack", exact: true }).click();
  107 |   await expect(page.getByLabel("Base model", { exact: true })).toHaveValue("fundamentals");
  108 |   await expect(state).toContainText("Ready");
  109 | });
  110 | 
  111 | test("named local views reload and restore the exact stack, race, scenario, and comparison", async ({ page }) => {
  112 |   await page.goto("/?run=history-2018-F-1&scenario=unresolved_other_R&race=2018%3AMN%3Asenate%3Aspecial&compare=1&against=history-2018-W-1&view=actual");
  113 |   await expandControls(page);
  114 |   await page.locator("details.composition-saved > summary").click();
  115 |   await page.getByLabel("Name this view", { exact: true }).fill("Midterm reference");
  116 |   await page.getByRole("button", { name: "Save view", exact: true }).click();
  117 |   await expect(page.getByRole("button", { name: "Restore Midterm reference", exact: true })).toBeVisible();
  118 |   const stored = await page.evaluate(() => JSON.parse(localStorage.getItem("ballotmancy.saved-views.v1") || "[]"));
  119 |   expect(stored).toHaveLength(1);
  120 |   expect(stored[0]).toMatchObject({ name: "Midterm reference", runId: "history-2018-F-1", scenarioId: "unresolved_other_R", raceId: "2018:MN:senate:special", compare: true, baselineRunId: "history-2018-W-1", mapView: "actual" });
  121 |   expect(stored[0].resultSha256).toMatch(/^[a-f0-9]{64}$/);
  122 |   await page.getByRole("group", { name: "Election year", exact: true }).getByRole("button", { name: "2026", exact: true }).click();
  123 |   await expect(page).toHaveURL(/run=current-/);
  124 |   await page.reload();
  125 |   await expandControls(page);
  126 |   await page.locator("details.composition-saved > summary").click();
  127 |   await page.getByRole("button", { name: "Restore Midterm reference", exact: true }).click();
  128 |   await expect(page).toHaveURL(/run=history-2018-F-1/);
  129 |   const restored = new URL(page.url()).searchParams;
  130 |   expect(restored.get("scenario")).toBe("unresolved_other_R");
  131 |   expect(restored.get("race")).toBe("2018:MN:senate:special");
  132 |   expect(restored.get("against")).toBe("history-2018-W-1");
  133 |   expect(restored.get("view")).toBe("actual");
  134 |   await expect(page.getByLabel("Comparison baseline", { exact: true })).toHaveValue("history-2018-W-1");
  135 |   await expect(page.getByRole("checkbox", { name: "Compare configuration", exact: true })).toBeChecked();
  136 |   await expect(page.getByLabel("Choose election in Minnesota", { exact: true })).toHaveValue("2018:MN:senate:special");
  137 |   await page.getByRole("button", { name: "Delete Midterm reference", exact: true }).click();
  138 |   await expect(page.getByRole("button", { name: "Restore Midterm reference", exact: true })).toHaveCount(0);
  139 | });
  140 | 
  141 | test("custom comparison identities survive URL reload and change probability comparisons", async ({ page }) => {
  142 |   await page.goto("/?run=history-2018-F-1");
  143 |   await expandControls(page);
  144 |   const baseline = page.getByLabel("Comparison baseline", { exact: true });
  145 |   await expect(page.getByRole("checkbox", {name:"Compare default",exact:true})).not.toBeChecked();
  146 |   await baseline.selectOption("history-2018-W-1");
  147 |   await expect(baseline).toHaveValue("history-2018-W-1");
  148 |   await expect(page.getByRole("checkbox", {name:"Compare configuration",exact:true})).not.toBeChecked();
  149 |   await expect(baseline.locator('option[value^="history-2022"]')).toHaveCount(0);
  150 |   const quality = page.getByRole("region", { name: "How this model did", exact: true }).getByRole("article", { name: "Probability quality", exact: true });
  151 |   await expect(quality).toContainText("higher error");
  152 |   await quality.getByText("Probability score details", { exact: true }).click();
  153 |   await expect(quality).toContainText("Average Brier score across races: 0.2418; comparison 0.2272");
  154 |   await baseline.selectOption("history-2018-F-1");
  155 |   await expect(page).toHaveURL(/against=history-2018-F-1/);
  156 |   await expect(quality).toContainText("Same error");
  157 |   await page.reload();
  158 |   await expandControls(page);
  159 |   await expect(baseline).toHaveValue("history-2018-F-1");
  160 |   await expect(quality).toContainText("Same error");
  161 | });
  162 | 
  163 | test("composition API binds ready presets and completed alternatives and rejects invalid requests without execution", async ({ request }) => {
  164 |   const response = await request.get("/api/configurations/catalog");
  165 |   expect(response.ok()).toBeTruthy();
  166 |   const catalog = await response.json() as CompositionCatalog;
  167 |   const preset = catalog.presets.find((entry) => entry.runId === "current-polling")!;
  168 |   const configuration = catalog.saved.find((entry) => entry.runId === preset.runId)!;
  169 |   const base = { catalogId: catalog.id, year: configuration.year, dateId: configuration.dateId, components: configuration.components };
  170 |   const readyResponse = await request.post("/api/configurations/resolve", { data: { ...base, presetId: preset.id } });
  171 |   expect(readyResponse.status()).toBe(200);
  172 |   const ready = await readyResponse.json();
  173 |   expect(ready.state).toBe("ready");
  174 |   expect(ready.configurationId).toBe(configuration.id);
  175 |   expect(ready.resultId).toBe(configuration.resultSha256);
  176 |   expect(ready.run.id).toBe(configuration.runId);
  177 |   const experimental = catalog.computation.find((entry) => entry.years.includes(configuration.year))!;
  178 |   const comparisonResponse = await request.post("/api/configurations/resolve", { data: { ...base, components: { ...base.components, base: experimental.baseId, polling: "standard", uncertainty: "standard" } } });
  179 |   expect(comparisonResponse.status()).toBe(200);
  180 |   expect(await comparisonResponse.json()).toMatchObject({ state: "ready", run: { id: `current-comparison-${experimental.baseId}-fp` }, executionAvailable: false });
  181 |   const incompatible = await request.post("/api/configurations/resolve", { data: { ...base, components: { ...base.components, polling: "none", uncertainty: "error-high" } } });
  182 |   expect(incompatible.status()).toBe(200);
  183 |   expect((await incompatible.json()).state).toBe("incompatible");
  184 |   const injected = await request.post("/api/configurations/resolve", { data: { ...base, bindings: { artifact: "client-chosen" } } });
  185 |   expect(injected.status()).toBe(400);
  186 |   const oversized = await request.post("/api/configurations/resolve", { headers: { "Content-Type": "application/json" }, data: " ".repeat(9000) });
  187 |   expect(oversized.status()).toBe(400);
  188 |   expect((await oversized.json()).error).toMatch(/too large/i);
  189 | });
  190 | 
```