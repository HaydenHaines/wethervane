import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { houseNy21Notice, houseNy21NoticeFor, houseNy21ReviewedResults } from "../lib/house-ny21-notice.ts";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const catalog = JSON.parse(readFileSync(join(root, "data/house-live/catalog.json"), "utf8"));
const stub = (entry, overrides = {}) => ({
  run: { id: entry.id, resultSha256: entry.result.sha256, ...overrides.run },
  races: [{ id: houseNy21Notice.raceId, district: { districtKey: "NY-21" } }, ...(overrides.races ?? [])],
});

test("the website erratum binds to exactly the sixteen published result identities", () => {
  assert.equal(catalog.entries.length, 16);
  assert.deepEqual(Object.entries(houseNy21ReviewedResults), catalog.entries.map(entry => [entry.id, entry.result.sha256]));
  for (const entry of catalog.entries) assert.equal(houseNy21NoticeFor(stub(entry)), houseNy21Notice, entry.id);
  const chosen = catalog.entries.find(entry => entry.id === catalog.default_id);
  assert.equal(chosen.result.sha256, "c84248d56ff0fd0e291538729680cf8cdaa51bad9fb456c3d359fa022b76451d");
});

test("replacement result hash, different run ID, or missing NY-21 never inherits the notice", () => {
  const entry = catalog.entries[0];
  assert.equal(houseNy21NoticeFor(stub(entry, { run: { resultSha256: "0".repeat(64) } })), null);
  assert.equal(houseNy21NoticeFor(stub(entry, { run: { id: "house-current-replacement" } })), null);
  assert.equal(houseNy21NoticeFor({ ...stub(entry), races: [] }), null);
  assert.equal(houseNy21NoticeFor({ ...stub(entry), races: [{ id: houseNy21Notice.raceId,
    district: { districtKey: "NY-20" } }] }), null);
});
