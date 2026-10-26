import assert from "node:assert/strict";
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import test from "node:test";
import { geoArea, geoContains, geoPath, geoAlbersUsa } from "d3-geo";
import { districtForProjection } from "../lib/district-projection.ts";

const square = (x, y, size) => [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]];
const feature = geometry => ({ type: "Feature", properties: { districtKey: "HI-02" }, geometry });
const hash = bytes => createHash("sha256").update(bytes).digest("hex");

test("mixed island winding preserves both islands and excludes the rest of the globe", () => {
  const source = feature({ type: "MultiPolygon", coordinates: [
    [square(-158, 20, 1)], [square(-156, 20, 1).reverse()],
  ] });
  const original = JSON.stringify(source);
  const projected = districtForProjection(source);
  assert.equal(geoContains(projected, [-157.5, 20.5]), true);
  assert.equal(geoContains(projected, [-155.5, 20.5]), true);
  assert.equal(geoContains(projected, [-100, 40]), false);
  assert.ok(geoArea(projected) > 0 && geoArea(projected) < .01);
  assert.equal(JSON.stringify(source), original);
  assert.deepEqual(districtForProjection(projected), projected);
});

test("holes keep their opposite orientation even when their input winding is wrong", () => {
  const source = feature({ type: "Polygon", coordinates: [square(-100, 35, 4), square(-99, 36, 1)] });
  const original = JSON.stringify(source);
  const projected = districtForProjection(source);
  assert.equal(geoContains(projected, [-99.5, 35.5]), true);
  assert.equal(geoContains(projected, [-98.5, 36.5]), false);
  assert.equal(geoContains(projected, [0, 0]), false);
  assert.equal(JSON.stringify(source), original);
});

test("actual published district geometry has no globe-complement or viewport-sized fills", (t) => {
  const root = new URL("../data/house-live/", import.meta.url);
  const manifestPath = new URL("publication.json", root);
  // Legacy branches can exercise the geometry mechanics without a current release.
  if (!existsSync(manifestPath)) return t.skip("No current release is staged on this branch.");
  const manifest = JSON.parse(readFileSync(manifestPath));
  const bytes = readFileSync(new URL(manifest.geography.artifact.path, root));
  assert.equal(hash(bytes), manifest.geography.artifact.sha256);
  const source = JSON.parse(bytes);
  const before = JSON.stringify(source);
  const display = { ...source, features: source.features.map(districtForProjection) };
  const path = geoPath(geoAlbersUsa().fitExtent([[22, 20], [938, 558]], display));
  assert.equal(display.features.length, 435);
  for (const district of display.features) {
    const id = district.properties.districtKey;
    const area = geoArea(district);
    assert.ok(area > 0 && area < 2 * Math.PI, `${id} must describe the district, not its spherical complement`);
    const bounds = path.bounds(district);
    assert.ok(bounds.flat().every(Number.isFinite), `${id} has finite projected bounds`);
    assert.ok(path.area(district) < 916 * 538 / 2, `${id} must not cover most of the map viewport`);
  }
  assert.equal(JSON.stringify(source), before);
});

test("all four historical district maps retain their source bytes and bounded fills", () => {
  const root = new URL("../data/house/", import.meta.url);
  const files = readdirSync(root).filter(name => /^geometry-[a-f0-9]{64}\.json$/.test(name));
  assert.equal(files.length, 4);
  for (const file of files) {
    const bytes = readFileSync(new URL(file, root));
    assert.equal(hash(bytes), file.slice(9, -5));
    const source = JSON.parse(bytes);
    const before = JSON.stringify(source);
    const display = { ...source, features: source.features.map(districtForProjection) };
    const path = geoPath(geoAlbersUsa().fitExtent([[22, 20], [938, 558]], display));
    for (const district of display.features) {
      const id = district.properties.districtKey;
      assert.ok(geoArea(district) > 0 && geoArea(district) < 2 * Math.PI, `${file}: ${id}`);
      assert.ok(path.bounds(district).flat().every(Number.isFinite), `${file}: ${id}`);
      assert.ok(path.area(district) < 916 * 538 / 2, `${file}: ${id}`);
    }
    assert.equal(JSON.stringify(source), before);
  }
});
