import test from "node:test";
import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadCountCatalogView } from "../lib/count-catalog-server.ts";
import { countCatalogPresentation } from "../lib/count-catalog-view.ts";
import { MODEL_SLICES } from "../lib/model-slices.ts";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const catalog = await loadCountCatalogView(join(root, "data/count-catalog"));

test("actual count metadata presents all eight required roles with exact provider identities", () => {
  const view = countCatalogPresentation(catalog);
  assert.ok(view);
  assert.deepEqual(view.roles.map(role => role.id), MODEL_SLICES.map(role => role.id));
  for (const role of view.roles) {
    const native = catalog.roles.find(row => row.role === role.id).providers[0];
    assert.equal(role.requiredLabel, "Required in this setup");
    assert.deepEqual(role.providers.map(p => [p.providerId, p.version, p.instanceId]), [[native.providerId, native.version, native.instanceId]]);
    assert.deepEqual(role.providers[0].options, []);
    assert.deepEqual(role.providers[0].limitations, native.limitations);
  }
  assert.match(view.roles.find(role => role.id === "measurement").label, /Surveys & participation reports/);
  assert.doesNotMatch(view.roles.find(role => role.id === "measurement").purpose, /polling choices|not connected/);
});

test("prediction target, training elections and input cutoff stay distinct", () => {
  const view = countCatalogPresentation(catalog);
  assert.equal(view.title, "Texas count-model research");
  assert.match(view.scopeLabel, /Prepared for the 2026 election/);
  assert.match(view.scopeLabel, /One Texas contest with a specified candidate list/);
  assert.equal(view.trainingLabel, "2020 and 2024");
  assert.equal(view.electionDate, "2026-11-03");
  assert.equal(view.informationCutoff, "2026-10-20T22:13:33Z");
  assert.match(view.informationCutoffLabel, /22:13:33 UTC/);
  assert.equal(view.technical.resolutionStatus, "requires_computation");
  assert.equal(view.technical.configurationSha256, catalog.prepared.configurationSha256);
});

test("four state descriptions distinguish preparation, absent result, accuracy evidence and private site", () => {
  const view = countCatalogPresentation(catalog);
  assert.deepEqual(view.states.map(state => state.id), ["preparation", "result", "evidence", "site"]);
  const text = view.states.map(state => `${state.value}. ${state.description}`).join(" ");
  assert.match(text, /All eight components are specified together/);
  assert.match(text, /No completed forecast supplied/);
  assert.match(text, /Forecast accuracy not assessed here/);
  assert.match(text, /Software checks alone do not establish forecast accuracy/);
  assert.match(text, /cannot schedule a calculation/);
  assert.doesNotMatch(text, /queued|running|no tests|untested|fit passed|accuracy verified/i);
});

test("role inventory ordering follows stable responsibilities rather than native array positions", () => {
  const reversed = structuredClone(catalog);
  reversed.roles.reverse();
  reversed.prepared.selections.reverse();
  assert.deepEqual(countCatalogPresentation(reversed), countCatalogPresentation(catalog));
});

test("unknown ready states, scheduler capability, missing roles and fabricated off choices fail closed", () => {
  for (const mutate of [
    view => { view.prepared.status = "ready"; },
    view => { view.statuses.scheduler = "available"; },
    view => { view.roles.pop(); },
    view => { view.prepared.selections[0].options = { mode: "off" }; },
    view => { view.scope.predictionCycle = 2018; },
    view => { view.prepared.independentRoleSwaps = true; },
  ]) {
    const input = structuredClone(catalog);
    mutate(input);
    assert.equal(countCatalogPresentation(input), null);
  }
});

test("presentation does not mutate or alias input metadata", () => {
  const original = structuredClone(catalog);
  const view = countCatalogPresentation(catalog);
  view.roles[0].providers[0].limitations.push("Presentation-only mutation");
  view.technical.bindings.catalogSha256 = "changed";
  assert.deepEqual(catalog, original);
});
