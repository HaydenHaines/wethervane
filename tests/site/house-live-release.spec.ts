/** Opt-in checks against an operator-staged, real 2026 House publication. No route mocks. */
import { createHash } from "node:crypto";
import { lstatSync, readFileSync, realpathSync } from "node:fs";
import { isAbsolute, join, relative, resolve } from "node:path";
import { expect, test, type Page } from "@playwright/test";

type Pin = { path: string; sha256: string };
type Configuration = { base: string; polling: "none" | "standard"; uncertainty: string };
type NativeCandidate = {
  candidate_id: string; name: string; party_label: string | null; aggregation_party: "D" | "R" | "O";
  ballot_status: "printed" | "declared_writein"; incumbent: boolean; source_ids: string[];
};
type NativePair = {
  first_candidate_id: string; second_candidate_id: string;
  denominator: string; mean: number;
  quantiles: { p025: number; p05: number; p10: number; p50: number; p90: number; p95: number; p975: number };
  p_first_exceeds_half: number; p_margin_within_2pp: number;
  p_first_margin_at_least_10pp: number; p_second_margin_at_least_10pp: number;
  qualifications: string[];
};
type NativeRace = {
  race_id: string; district_key: string; poll_count: number; poll_source_ids: string[];
  state: string; map_vintage: string; rule_family: string; method: string;
  qualifications: string[]; candidates: NativeCandidate[]; conditional_shares: NativePair | null;
  outcomes: { outcome_id: string; label: string; party: "D" | "R" | "O";
    kind: "candidate" | "group"; member_candidate_ids: string[]; p_win: number }[];
};
type NativeResult = {
  id: string; cycle: number; configuration: Configuration; map_vintage: string;
  issued_at: string; forecast_cutoff: string; election_date: string;
  source_cutoffs: { role: string; at: string }[]; strict_as_of: boolean;
  provenance: { id: string; label: string; url: string; retrieved_at: string; sha256: string; role: string;
    capture_kind?: "received" | "rendered_primary_source"; limitations?: string[] }[];
  historical_evidence: { label: string; url: string; qualification: string }[]; races: NativeRace[];
  joint_draws: { sha256: string; count: number; chain_count: number; ordered_race_ids: string[]; identity_semantics: string };
  chamber: null | { majority: { D: number; R: number; unresolved: number };
    precision: { max_seat_mean_mcse: number; max_control_mcse: number };
    seats: { D: { expected: number; quantiles: { p05: number; p50: number; p95: number } };
      R: { expected: number; quantiles: { p05: number; p50: number; p95: number } };
      O: { expected: number; quantiles: { p05: number; p50: number; p95: number } } } };
  qualification: { input_status: string; numerical_status: string; limitations: string[];
    publication_status: string; forecast_skill: string; evidence: { sha256: string }[] };
};
type CatalogEntry = { id: string; label: string; configuration: Configuration; result: Pin };
type Catalog = { schema: string; default_id: string; entries: CatalogEntry[] };
type Manifest = {
  schema: string; status: string; cycle: number; catalog: Pin; review: Pin;
  pollingDisclosure?: Pin; results: { id: string; artifact: Pin }[];
};
type PollingDisclosure = {
  schema: string; cycle: number; ruleset: string;
  cutoffs: { forecastAt: string; pollingAt: string };
  bindings: { catalogSha256: string; sourcePolicySha256: string; admissionSha256: string;
    selectionSha256: string; ledgerSha256: string };
  selected: PollingQuestion[]; excluded: PollingExcluded[];
  excludedReasonCounts: { districtKey: string | null; reason: string; count: number }[];
  runs: { id: string; resultSha256: string; polling: string }[];
};
type PollingQuestion = { observationId: string; raceId: string; pollster: string;
  startDate: string; fieldEnd: string; availableAt: string; population: "LV" | "RV" | "A";
  sampleSize: number | null; sourceId: string; sourceSha256: string; sourceUrls: string[] };
type PollingExcluded = { districtKey: string | null; pollster: string; endDate: string | null; reason: string };
type Release = {
  manifest: Manifest; catalog: Catalog; review: { decision: string; limitations: string[];
    catalog_sha256: string; results: { id: string; result_sha256: string; joint_draws_sha256: string }[] };
  results: Map<string, NativeResult>; polling: PollingDisclosure;
};

const enabled = process.env.BALLOTMANCY_LIVE_HOUSE_RELEASE === "1";
const directory = resolve(process.cwd(), "data/house-live");
let release: Release;

function readPinned<T>(pin: Pin): T {
  if (!/^[a-f0-9]{64}$/.test(pin.sha256) || !/^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(pin.path) ||
    pin.path.split("/").some(part => !part || part === "." || part === ".."))
    throw Error(`Unsafe staged House artifact pin: ${pin.path}`);
  const root = realpathSync(directory);
  const path = join(root, pin.path);
  const actual = realpathSync(path);
  const inside = relative(root, actual);
  if (!inside || inside.startsWith("..") || isAbsolute(inside) || !lstatSync(actual).isFile())
    throw Error(`Staged House artifact leaves its release directory: ${pin.path}`);
  const bytes = readFileSync(actual);
  const digest = createHash("sha256").update(bytes).digest("hex");
  if (digest !== pin.sha256) throw Error(`Staged House artifact hash changed: ${pin.path}`);
  return JSON.parse(bytes.toString("utf8")) as T;
}

function loadRelease(): Release {
  // Enabled mode deliberately fails if this file or any required staged pin is absent.
  const manifestPath = join(directory, "publication.json");
  if (!lstatSync(manifestPath).isFile() || realpathSync(manifestPath) !== manifestPath)
    throw Error("Live House publication.json must be a regular staged file.");
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Manifest;
  if (manifest.schema !== "HouseLiveSiteManifestV1" || manifest.status !== "available" ||
    manifest.cycle !== 2026 || manifest.results?.length !== 16 || !manifest.pollingDisclosure)
    throw Error("The real 2026 House release is absent, incomplete, or lacks polling disclosure.");
  const catalog = readPinned<Catalog>(manifest.catalog);
  const review = readPinned<Release["review"]>(manifest.review);
  const polling = readPinned<PollingDisclosure>(manifest.pollingDisclosure);
  if (catalog.schema !== "HouseLiveForecastCatalogV1" || catalog.entries?.length !== 16 ||
    catalog.default_id !== "house-current-polling" || review.decision !== "publish" ||
    review.catalog_sha256 !== manifest.catalog.sha256 || review.results?.length !== 16 ||
    polling.schema !== "HouseLiveSitePollingDisclosureV1" || polling.runs?.length !== 16)
    throw Error("The staged House catalog, review, or polling disclosure is malformed.");
  const results = new Map<string, NativeResult>();
  for (const entry of catalog.entries) {
    const staged = manifest.results.find(row => row.id === entry.id);
    const reviewed = review.results.find(row => row.id === entry.id);
    const pollRun = polling.runs.find(row => row.id === entry.id);
    if (!staged || !reviewed || staged.artifact.sha256 !== entry.result.sha256 ||
      reviewed.result_sha256 !== entry.result.sha256 || !pollRun ||
      pollRun.resultSha256 !== entry.result.sha256 || pollRun.polling !== entry.configuration.polling)
      throw Error(`The staged House release does not bind ${entry.id}.`);
    const result = readPinned<NativeResult>(entry.result);
    if (result.id !== entry.id || result.qualification.publication_status !== "reviewed" ||
      result.races?.length !== 435 || result.joint_draws?.ordered_race_ids?.length !== 435 ||
      reviewed.joint_draws_sha256 !== result.joint_draws.sha256)
      throw Error(`The staged House result is incomplete or unreviewed: ${entry.id}.`);
    results.set(entry.id, result);
  }
  if (results.size !== 16 || !results.has(catalog.default_id)) throw Error("The staged House roster is incomplete.");
  return { manifest, catalog, review, results, polling };
}

const native = (id: string) => {
  const result = release.results.get(id);
  if (!result) throw Error(`Missing authenticated native result ${id}.`);
  return result;
};
const configKey = (value: Configuration) => `${value.base}|${value.polling}|${value.uncertainty}`;
const probability = (value: number) => value > 0 && value < .001 ? "<0.1%"
  : value < 1 && value > .999 ? ">99.9%" : `${(value * 100).toFixed(1)}%`;
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const rgb = (hex: string) => `rgb(${[1, 3, 5].map(at => parseInt(hex.slice(at, at + 2), 16)).join(", ")})`;
const outlook = (result: NativeResult) => {
  const control = result.chamber?.majority;
  if (!control) return { party: "unavailable", color: "" };
  if (control.D > .6) return { party: "D", color: rgb("#3e6683") };
  if (control.R > .6) return { party: "R", color: rgb("#a3553f") };
  if (control.unresolved > .6) return { party: "U", color: rgb("#726e66") };
  return { party: "tossup", color: rgb("#7b658e") };
};
function districtColor(race: NativeRace): string {
  const leader = [...race.outcomes].sort((a, b) => b.p_win - a.p_win)[0];
  if (leader.p_win <= .6) return "#7b658e";
  const colors = leader.p_win > .9
    ? { D: "#3e6683", R: "#a3553f", O: "#726e66" }
    : { D: "#92afc1", R: "#d4a28c", O: "#beb9ad" };
  return colors[leader.party];
}
async function openBuilder(page: Page) {
  const panel = page.locator("details.house-configuration");
  if (await panel.getAttribute("open") === null) await panel.locator(":scope > summary").click();
  return panel;
}

test.describe("actual staged 2026 House release", () => {
  test.skip(!enabled, "Set BALLOTMANCY_LIVE_HOUSE_RELEASE=1 only after staging the reviewed real release.");
  test.beforeAll(() => { release = loadRelease(); });

  test("availability and all 16 API choices match authenticated native results", async ({ request }) => {
    test.setTimeout(180_000);
    const availability = await request.get("/api/chambers/house/live/availability");
    expect(availability.status()).toBe(200);
    expect(await availability.json()).toEqual({ schema: "HouseLiveAvailabilityV1", available: true });
    for (const entry of release.catalog.entries) {
      const response = await request.get(`/api/chambers/house/live?run=${encodeURIComponent(entry.id)}`);
      expect(response.status(), entry.id).toBe(200);
      const publication = await response.json();
      const result = native(entry.id);
      expect(publication.status).toBe("available");
      expect(publication.catalog.defaultId).toBe(release.catalog.default_id);
      expect(publication.catalog.entries).toHaveLength(16);
      expect(publication.view.run).toMatchObject({ id: entry.id, resultSha256: entry.result.sha256,
        configuration: entry.configuration, cycle: result.cycle, historical: false,
        issuedAt: result.issued_at, cutoffAt: result.forecast_cutoff, electionDate: result.election_date,
        strictAsOf: result.strict_as_of, sourceCutoffs: result.source_cutoffs });
      expect(configKey(publication.view.run.configuration)).toBe(configKey(result.configuration));
      expect(publication.view.races).toHaveLength(435);
      expect(publication.view.jointDraws).toMatchObject({ sha256: result.joint_draws.sha256,
        count: result.joint_draws.count, chainCount: result.joint_draws.chain_count,
        orderedRaceIds: result.joint_draws.ordered_race_ids,
        identitySemantics: result.joint_draws.identity_semantics });
      expect(publication.view.mapVintage).toBe(result.map_vintage);
      expect(publication.view.qualification).toEqual({
        inputStatus: result.qualification.input_status, numericalStatus: result.qualification.numerical_status,
        publicationStatus: result.qualification.publication_status, forecastSkill: result.qualification.forecast_skill,
        limitations: result.qualification.limitations,
        evidenceHashes: result.qualification.evidence.map(pin => pin.sha256),
      });
      expect(publication.view.sources).toEqual(result.provenance.map(
        ({ id, label, url, retrieved_at, sha256, role, capture_kind, limitations }) => ({
          id, label, url, retrieved_at, sha256, role,
          ...(capture_kind === undefined ? {} : { capture_kind }),
          ...(limitations === undefined ? {} : { limitations }),
        })));
      expect(publication.view.historicalEvidence).toEqual(result.historical_evidence);
      expect(publication.publicationReview?.limitations).toEqual(release.review.limitations);
      if (result.chamber) {
        expect(publication.view.chamber.control).toMatchObject({
          democraticMajorityProbability: result.chamber.majority.D,
          republicanMajorityProbability: result.chamber.majority.R,
          unresolvedOrganizingProbability: result.chamber.majority.unresolved,
        });
        for (const [nativeParty, displayParty] of [["D", "D"], ["R", "R"], ["O", "U"]] as const)
          expect(publication.view.chamber.seats[displayParty]).toMatchObject({
            expectedSeats: result.chamber.seats[nativeParty].expected,
            seatQuantiles: {
              p05: result.chamber.seats[nativeParty].quantiles.p05,
              p50: result.chamber.seats[nativeParty].quantiles.p50,
              p95: result.chamber.seats[nativeParty].quantiles.p95,
            },
          });
        expect(publication.view.chamberPrecision).toMatchObject({
          maxSeatMeanMcse: result.chamber.precision.max_seat_mean_mcse,
          maxControlMcse: result.chamber.precision.max_control_mcse,
        });
      } else expect(publication.view.chamber).toBeNull();
      expect(publication.view.races.map((race: { id: string }) => race.id)).toEqual(result.races.map(race => race.race_id));
      const sources = new Map(result.provenance.map(source => [source.id, source.url]));
      for (let index = 0; index < 435; index++) {
        const observed = publication.view.races[index];
        const source = result.races[index];
        const urls = (ids: string[]) => [...new Set(ids.map(id => {
          const url = sources.get(id);
          if (!url) throw Error(`Native source ID ${id} is missing from ${entry.id} provenance.`);
          return url;
        }))];
        const candidates = new Map(source.candidates.map(candidate => [candidate.candidate_id, candidate]));
        expect(observed).toMatchObject({
          id: source.race_id, state: source.state, seatId: source.district_key,
          electionLabel: "2026 House election", pollCount: source.poll_count,
          notes: source.qualifications,
          district: { cycle: result.cycle, districtKey: source.district_key, mapVintage: source.map_vintage },
          ruleFamily: source.rule_family, method: source.method,
        });
        expect(observed.candidates).toEqual(source.candidates.map(candidate => ({
          id: candidate.candidate_id, name: candidate.name, partyLabel: candidate.party_label,
          aggregationParty: candidate.aggregation_party, ballotStatus: candidate.ballot_status,
          incumbent: candidate.incumbent, sourceUrls: urls(candidate.source_ids),
        })));
        expect(observed.outcomes).toEqual(source.outcomes.map(outcome => ({
          id: `${source.race_id}:${outcome.outcome_id}`, name: outcome.label, party: outcome.party,
          kind: outcome.kind, winProbability: outcome.p_win,
          members: outcome.member_candidate_ids.map(id => {
            const candidate = candidates.get(id);
            if (!candidate) throw Error(`Native outcome member ${id} is missing from ${source.race_id}.`);
            return { id, name: candidate.name, party: candidate.aggregation_party,
              incumbent: candidate.incumbent, sourceUrls: urls(candidate.source_ids) };
          }),
        })));
        const pair = source.conditional_shares;
        expect(observed.conditionalPair).toEqual(pair ? {
          firstCandidateId: pair.first_candidate_id, secondCandidateId: pair.second_candidate_id,
          firstCandidateName: candidates.get(pair.first_candidate_id)!.name,
          secondCandidateName: candidates.get(pair.second_candidate_id)!.name,
          denominator: pair.denominator, mean: pair.mean, quantiles: pair.quantiles,
          firstExceedsHalf: pair.p_first_exceeds_half, marginWithinTwoPoints: pair.p_margin_within_2pp,
          firstMarginAtLeastTenPoints: pair.p_first_margin_at_least_10pp,
          secondMarginAtLeastTenPoints: pair.p_second_margin_at_least_10pp,
          qualifications: pair.qualifications,
        } : null);
      }
      const pollRun = release.polling.runs.find(row => row.id === entry.id);
      expect(pollRun).toMatchObject({ id: entry.id, resultSha256: entry.result.sha256,
        polling: entry.configuration.polling });
      const expectedPollQuestions = entry.configuration.polling === "standard"
        ? release.polling.selected : [];
      const nativePollIds = result.races.flatMap(race => race.poll_source_ids).sort();
      expect(expectedPollQuestions.map(row => row.observationId).sort(), entry.id).toEqual(nativePollIds);
      expect(publication.polling).toEqual({
        schema: "HouseLivePollingRunV1", runId: entry.id, resultSha256: entry.result.sha256,
        ruleset: release.polling.ruleset,
        cutoffs: release.polling.cutoffs, bindings: release.polling.bindings,
        modelUsesSelectedQuestions: entry.configuration.polling === "standard",
        usedQuestions: expectedPollQuestions,
        selectedPoolQuestionCount: release.polling.selected.length,
        selectedPoolDistrictCount: new Set(release.polling.selected.map(row => row.raceId)).size,
        excluded: release.polling.excluded, excludedReasonCounts: release.polling.excludedReasonCounts,
      });
      expect(JSON.stringify(publication)).not.toMatch(/\/home\/|file:\/\//);
    }
  });

  test("the real default renders its native chamber, all districts, and review notes", async ({ page }) => {
    test.setTimeout(120_000);
    const result = native(release.catalog.default_id);
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto("/?chamber=house");
    await expect(page.getByRole("heading", { name: /The House,\s*in perspective\./ })).toBeVisible();
    await expect(page).toHaveURL(new RegExp(`run=${release.catalog.default_id}`));
    const heading = page.locator(".hero .chamber-name");
    const expected = outlook(result);
    await expect(heading).toHaveAttribute("data-chamber-outlook", expected.party);
    if (expected.color) await expect(heading).toHaveCSS("color", expected.color);
    await expect(page.locator(".district-map-svg [data-forecast=available]")).toHaveCount(435);
    await expect(page.getByRole("region", { name: "House group-win probability ribbon" })
      .locator("button.house-ribbon-cell")).toHaveCount(435);
    await expect(page.locator("#house-live-race-list > button")).toHaveCount(10);
    await expect(page.locator("details.house-configuration")).not.toHaveAttribute("open", "");
    await page.locator('[aria-controls="house-live-race-list"]').click();
    await expect(page.locator("#house-live-race-list > button")).toHaveCount(435);
    await page.locator('[aria-controls="house-live-race-list"]').click();
    await expect(page.locator("#house-live-race-list > button")).toHaveCount(10);
    if (result.chamber) {
      await expect(page.locator("#house-live-control-title")).toBeVisible();
      await expect(page.locator(".control-probs .prob-number")).toHaveText([
        probability(result.chamber.majority.D), probability(result.chamber.majority.R),
        probability(result.chamber.majority.unresolved),
      ]);
      await expect(page.locator(".chamber-seat-summary b")).toHaveText([
        result.chamber.seats.D.expected.toFixed(1), result.chamber.seats.R.expected.toFixed(1),
        result.chamber.seats.O.expected.toFixed(1),
      ]);
      await expect(page.locator(".chamber-seat-summary .small")).toHaveText([
        `(${result.chamber.seats.D.quantiles.p05}–${result.chamber.seats.D.quantiles.p95}, modeled 90% range)`,
        `(${result.chamber.seats.R.quantiles.p05}–${result.chamber.seats.R.quantiles.p95}, modeled 90% range)`,
        `(${result.chamber.seats.O.quantiles.p05}–${result.chamber.seats.O.quantiles.p95}, modeled 90% range)`,
      ]);
    }
    for (const note of result.qualification.limitations) await expect(page.locator("#methodology")).toContainText(note);
    const captureNotes = result.provenance.filter(source => source.limitations?.length);
    if (captureNotes.length) {
      const disclosure = page.locator("#methodology details").filter({ hasText: "Source capture notes" });
      await expect(disclosure).not.toHaveAttribute("open", "");
      await disclosure.locator(":scope > summary").click();
      for (const source of captureNotes) {
        await expect(disclosure.getByRole("link", { name: source.label, exact: true })).toHaveAttribute("href", source.url);
        for (const note of source.limitations!) await expect(disclosure).toContainText(note);
      }
      await disclosure.locator(":scope > summary").click();
    }
    if (release.review.limitations.length) {
      await expect(page.getByRole("heading", { name: "Publication review notes" })).toBeVisible();
      for (const note of release.review.limitations) await expect(page.locator("#methodology")).toContainText(note);
    }
    const builder = await openBuilder(page);
    await expect(builder.getByLabel("Electoral-history model")).toHaveValue(result.configuration.base);
    await expect(builder.getByLabel("Uncertainty scenario")).toHaveValue(result.configuration.uncertainty);
    await expect(builder.getByRole("checkbox", { name: "Include current polling evidence" }))
      .toBeChecked({ checked: result.configuration.polling === "standard" });
    expect(errors).toEqual([]);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  });

  test("one renderer matches every real model's district colors, ribbon, and control values", async ({ page }) => {
    test.setTimeout(240_000);
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    for (const entry of release.catalog.entries) {
      const result = native(entry.id);
      await page.goto(`/?chamber=house&edition=current&run=${encodeURIComponent(entry.id)}`);
      await expect(page.locator(".hero")).toContainText(entry.label);
      await expect(page.locator(".district-map-svg [data-forecast=available]")).toHaveCount(435);
      const tiles = page.getByRole("region", { name: "House group-win probability ribbon" })
        .locator("button.house-ribbon-cell");
      await expect(tiles).toHaveCount(435);
      const mapColors = await page.locator(".district-map-svg [data-forecast=available]").evaluateAll(nodes =>
        Object.fromEntries(nodes.map(node => [node.getAttribute("data-district-key"), node.getAttribute("fill")])));
      const ribbonColors = await tiles.evaluateAll(nodes => Object.fromEntries(nodes.map(node =>
        [node.getAttribute("data-district"), getComputedStyle(node).backgroundColor])));
      expect(mapColors, entry.id).toEqual(Object.fromEntries(result.races.map(race =>
        [race.district_key, districtColor(race)])));
      const oversizedFills = await page.locator(".district-map-svg [data-forecast=available]").evaluateAll(nodes =>
        nodes.filter(node => {
          const path = node as SVGPathElement, box = path.getBBox();
          const viewport = path.ownerSVGElement!.viewBox.baseVal;
          return ![box.width, box.height].every(Number.isFinite) || box.width <= 0 || box.height <= 0 ||
            box.width >= viewport.width / 2 || box.height >= viewport.height / 2;
        }).map(node => node.getAttribute("data-district-key")));
      expect(oversizedFills, `${entry.id}: a district must not fill most of the map`).toEqual([]);
      expect(ribbonColors, entry.id).toEqual(Object.fromEntries(result.races.map(race =>
        [race.district_key, rgb(districtColor(race))])));
      await expect(page.locator(".hero .chamber-name")).toHaveAttribute("data-chamber-outlook", outlook(result).party);
      if (result.chamber) await expect(page.locator(".control-probs .prob-number")).toHaveText([
        probability(result.chamber.majority.D), probability(result.chamber.majority.R),
        probability(result.chamber.majority.unresolved),
      ]);
      else await expect(page.getByRole("region", { name: "House control availability" })).toBeVisible();
      // Independently count 218 from each end of the supplied D/R path. Other
      // leaders/ties never become seats for either party or a tipping statistic.
      const ranked = result.races.map(race => {
        const p = (party: string) => race.outcomes.filter(outcome => outcome.party === party)
          .reduce((sum, outcome) => sum + outcome.p_win, 0);
        return { district: race.district_key, D: p("D"), R: p("R"), O: p("O") };
      }).filter(row => row.O < Math.max(row.D, row.R))
        .sort((a, b) => (b.D - b.R) - (a.D - a.R) || a.district.localeCompare(b.district));
      const expectedMarkers = [...new Set([ranked[217].district, ranked[ranked.length - 218].district])];
      const markers = page.locator(".house-ribbon-cell.majority-position");
      await expect(markers).toHaveCount(expectedMarkers.length);
      expect((await markers.evaluateAll(nodes => nodes.map(node => node.getAttribute("data-district")))).sort())
        .toEqual(expectedMarkers.sort());
      await expect(page.locator(".house-majority-summary")).toContainText("218 seats");
      await expect(page.locator(".house-tipping-summary")).toHaveCount(0);
      for (const marker of await markers.all()) {
        await expect(marker).toHaveAccessibleName(/majority marker, 218 seats/);
        const bar = await marker.evaluate(node => {
          const style = getComputedStyle(node, "::after");
          return { content: style.content, height: parseFloat(style.height), width: parseFloat(style.borderLeftWidth) };
        });
        expect(bar.content).not.toBe("none"); expect(bar.height).toBeGreaterThan(0); expect(bar.width).toBeGreaterThan(0);
      }
      if (entry.id === release.catalog.default_id) {
        const district = expectedMarkers[0];
        await page.locator(".house-majority-summary").getByRole("button", { name: district, exact: true }).click();
        await expect(page.locator("#house-live-detail-title")).toContainText(district);
        await page.getByLabel("Show district labels and larger tiles").check();
        await expect(markers.first()).toHaveAttribute("data-district", district);
        await expect(markers.first()).toBeVisible();
      }
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), entry.id).toBe(true);
    }
    expect(errors).toEqual([]);
  });

  test("real polling, zero-local and mixed Other districts survive selection and comparison", async ({ page }, testInfo) => {
    test.setTimeout(150_000);
    const onEntry = release.catalog.entries.find(entry => entry.configuration.base === "fundamentals" &&
      entry.configuration.polling === "standard" && entry.configuration.uncertainty === "standard")!;
    const offEntry = release.catalog.entries.find(entry => entry.configuration.base === "fundamentals" &&
      entry.configuration.polling === "none" && entry.configuration.uncertainty === "standard")!;
    expect(onEntry).toBeTruthy(); expect(offEntry).toBeTruthy();
    const on = native(onEntry.id);
    const withPoll = on.races.find(race => race.poll_count > 0);
    const zeroLocal = on.races.find(race => race.poll_count === 0);
    const supportedPair = on.races.find(race => race.conditional_shares !== null);
    const mixedOther = on.races.find(race => race.outcomes.some(row => row.party === "O" && row.kind === "group" &&
      row.member_candidate_ids.length > 1 && new Set(row.member_candidate_ids.map(id =>
        race.candidates.find(candidate => candidate.candidate_id === id)?.party_label).filter(Boolean)).size > 1));
    expect(withPoll).toBeTruthy(); expect(zeroLocal).toBeTruthy(); expect(supportedPair).toBeTruthy();
    await page.goto(`/?chamber=house&edition=current&run=${onEntry.id}&race=${encodeURIComponent(withPoll!.race_id)}`);
    await expect(page.locator("#house-live-detail-title")).toHaveText(withPoll!.district_key);
    await expect(page.locator("#house-live-race-detail")).toContainText(`${withPoll!.poll_count} selected local poll-question`);
    await page.locator("#house-live-polling .poll-records > summary").click();
    await expect(page.locator(`#house-live-polling [aria-label="Included poll questions for ${withPoll!.district_key}"] > li`))
      .toHaveCount(withPoll!.poll_count);
    const builder = await openBuilder(page);
    await expect(builder.getByRole("checkbox", { name: "Include current polling evidence" })).toBeChecked();
    await builder.getByRole("checkbox", { name: "Include current polling evidence" }).uncheck();
    await expect(page).toHaveURL(new RegExp(`run=${offEntry.id}`));
    await expect(page.locator("#house-live-polling")).toContainText("Polling is switched off.");
    await expect(page.locator("#house-live-detail-title")).toHaveText(withPoll!.district_key);
    await page.locator("#house-live-comparison").selectOption(onEntry.id);
    await expect(page.getByRole("region", { name: "Current House model comparison" }))
      .toContainText("Selected district");
    await expect(page.getByRole("region", { name: "Current House model comparison" }))
      .toContainText(withPoll!.district_key);
    await page.goto(`/?chamber=house&edition=current&run=${onEntry.id}&race=${encodeURIComponent(zeroLocal!.race_id)}`);
    await page.locator("#house-live-polling .poll-records > summary").click();
    await expect(page.locator("#house-live-polling")).toContainText(`No eligible local poll question was selected for ${zeroLocal!.district_key}`);
    if (mixedOther) {
      await page.goto(`/?chamber=house&edition=current&run=${onEntry.id}&race=${encodeURIComponent(mixedOther.race_id)}`);
      const detail = page.locator("#house-live-race-detail");
      await expect(detail.locator(".candidate")).toHaveCount(mixedOther.outcomes.length);
      for (const [index, outcome] of mixedOther.outcomes.entries()) {
        if (outcome.party !== "O" || outcome.kind !== "group") continue;
        const row = detail.locator(".candidate").nth(index);
        await expect(row.locator(".party-label")).toHaveText("Other (O) · Group outcome");
        for (const id of outcome.member_candidate_ids) {
          const member = mixedOther.candidates.find(candidate => candidate.candidate_id === id)!;
          await expect(row).toContainText(member.name);
          if (member.party_label) await expect(row).toContainText(member.party_label);
        }
      }
    } else testInfo.annotations.push({ type: "coverage", description:
      "No real Other group contains differently labeled members in this release; synthetic mixed-group coverage remains separate." });
    const pairRace = supportedPair!;
    await page.goto(`/?chamber=house&edition=current&run=${onEntry.id}&race=${encodeURIComponent(pairRace.race_id)}`);
    const detail = page.locator("#house-live-race-detail");
    await expect(detail.locator(".candidate-prob")).toHaveText(pairRace.outcomes.map(outcome => probability(outcome.p_win)));
    const pair = pairRace.conditional_shares!;
    const pairView = detail.getByRole("region", { name: "Conditional two-candidate vote share" });
    await expect(pairView).toContainText(`Pair: ${pairRace.candidates.find(candidate => candidate.candidate_id === pair.first_candidate_id)!.name} and ${pairRace.candidates.find(candidate => candidate.candidate_id === pair.second_candidate_id)!.name}.`);
    await expect(pairView).toContainText(`First candidate average: ${percent(pair.mean)}. Denominator: ${pair.denominator}.`);
    await expect(pairView).toContainText(`Middle 90% of modeled shares: ${percent(pair.quantiles.p05)}–${percent(pair.quantiles.p95)}.`);
    await expect(pairView).toContainText(`First candidate exceeds half: ${probability(pair.p_first_exceeds_half)}.`);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  });
  test("the reviewed NY-21 erratum leads from the forecast to the affected race and linked sources", async ({ page, request }, testInfo) => {
    test.setTimeout(120_000);
    for (const item of release.catalog.entries) {
      const response = await request.get(`/api/chambers/house/live?run=${encodeURIComponent(item.id)}`);
      expect(response.status(), item.id).toBe(200);
      const publication = await response.json();
      expect(publication.view.run.resultSha256, item.id).toBe(item.result.sha256);
      const shown = publication.view.races.find((race: { id: string }) => race.id === "2026:NY-21:house:regular");
      const original = native(item.id).races.find(race => race.race_id === "2026:NY-21:house:regular");
      expect(shown, item.id).toBeTruthy(); expect(original, item.id).toBeTruthy();
      expect(shown.outcomes.map((outcome: { winProbability: number }) => outcome.winProbability), item.id)
        .toEqual(original!.outcomes.map(outcome => outcome.p_win));
    }
    const entry = release.catalog.entries.find(item => item.id === release.catalog.default_id)!;
    expect(entry.result.sha256).toBe("c84248d56ff0fd0e291538729680cf8cdaa51bad9fb456c3d359fa022b76451d");
    await page.goto("/?chamber=house&edition=current");
    const notice = page.getByRole("region", { name: "NY-21 forecast input notice" });
    await expect(notice).toContainText("NY-21 is under review. The October 25 forecast omits declared write-in candidate Christopher Schmidt and excludes two polls that name him.");
    await notice.getByRole("button", { name: "View NY-21" }).click();
    const detailHeading = page.locator("#house-live-detail-title");
    await expect(detailHeading).toHaveText("NY-21");
    await expect(detailHeading).toBeFocused();
    await expect.poll(() => detailHeading.evaluate(node => {
      const box = node.getBoundingClientRect();
      return box.top >= 0 && box.bottom <= innerHeight;
    })).toBe(true);
    const focus = page.getByRole("region", { name: "NY-21 forecast input detail" });
    await expect(focus).toBeVisible();
    await expect(focus.getByRole("link")).toHaveCount(3);
    await expect(focus.getByRole("link", { name: /New York candidate list/ })).toBeVisible();
    await expect(focus.getByRole("link", { name: /NCPR: write-in campaign/ })).toBeVisible();
    await expect(focus.getByRole("link", { name: /Siena poll/ })).toBeVisible();
    await expect(focus).toContainText("The displayed probabilities remain the October 25 estimates.");
    await expect(focus).not.toContainText("source-coverage erratum");
    await expect(page.getByRole("region", { name: "NY-21 input review" })).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath("ny21-notice.png"), fullPage: true });
    await page.goto("/?chamber=house&edition=current&race=2026%3AAL-01%3Ahouse%3Aregular");
    await expect(notice).toBeVisible();
    await expect(page.getByRole("region", { name: "NY-21 forecast input detail" })).toHaveCount(0);
    await page.goto("/?chamber=house&edition=current&run=house-current-fundamentals&race=2026%3ANY-21%3Ahouse%3Aregular");
    await expect(notice).toBeVisible();
    await expect(page.getByRole("region", { name: "NY-21 forecast input detail" })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  });
});
