import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";

const geometryBytes = readFileSync("data/house-live/geography/great-lakes-display.json");
const receipt = JSON.parse(readFileSync("data/house-live/geography/great-lakes-display.receipt.json", "utf8"));
const geometry = JSON.parse(geometryBytes.toString("utf8"));
const digest = bytes => createHash("sha256").update(bytes).digest("hex");

function ringContains(ring, [x, y]) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function polygonContains(rings, point) {
  return ringContains(rings[0], point) && !rings.slice(1).some(ring => ringContains(ring, point));
}
function featureContains(feature, point) {
  const { type, coordinates } = feature.geometry;
  return type === "Polygon"
    ? polygonContains(coordinates, point)
    : type === "MultiPolygon" && coordinates.some(polygon => polygonContains(polygon, point));
}

test("Natural Earth Great Lakes display mask is pinned and has lake-water-only coverage", () => {
  assert.equal(receipt.schema, "HouseGreatLakesDisplayGeometryV1");
  assert.equal(receipt.purpose.startsWith("Display-only"), true);
  assert.equal(receipt.sourceVersion, "5.0.0");
  assert.equal(receipt.sourceArchiveSha256, "0803a06f9c3cb4671d89b68c48b142aad9366ba40f665245e12a913fbc61722a");
  assert.equal(receipt.sha256, digest(geometryBytes));
  assert.equal(receipt.sha256, "0d6b9b01862ccb959d815b2568cde309ae4623f0cdfce542f3d97186529e3e4e");
  assert.equal(geometry.type, "FeatureCollection");
  assert.deepEqual(geometry.features.map(feature => feature.properties.name), [
    "Lake Erie", "Lake Huron", "Lake Michigan", "Lake Ontario", "Lake Superior",
  ]);

  for (const [name, point] of [
    ["Lake Superior", [-87, 47]], ["Lake Michigan", [-87, 44]],
    ["Lake Huron", [-82.5, 45]], ["Lake Erie", [-81.5, 42]], ["Lake Ontario", [-77.7, 43.7]],
  ]) {
    const lake = geometry.features.find(feature => feature.properties.name === name);
    assert.ok(featureContains(lake, point), `${name} water point must be masked`);
  }
  const land = [-85.6, 42.8];
  assert.equal(geometry.features.some(feature => featureContains(feature, land)), false, "nearby Michigan land must remain exposed");
});
