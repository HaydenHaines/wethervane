import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GET as forecast } from "../app/api/chambers/house/live/route.ts";
import { GET as availability } from "../app/api/chambers/house/live/availability/route.ts";

test("live API distinguishes absent publication, invalid selection and broken release without leaking file details", async () => {
  const directory = await mkdtemp(join(tmpdir(), "house-live-api-"));
  const original = process.cwd(), originalError = console.error;
  const serverErrors = [];
  try {
    process.chdir(directory);
    console.error = (...args) => serverErrors.push(args);
    const get = (handler, query = "") => handler(new Request(`http://localhost/api${query}`));
    const absent = await get(availability);
    assert.equal(absent.status, 200);
    assert.equal(absent.headers.get("cache-control"), "no-store");
    assert.deepEqual(await absent.json(), { schema: "HouseLiveAvailabilityV1", available: false });
    const missing = await get(forecast);
    assert.equal(missing.status, 503);
    assert.doesNotMatch(JSON.stringify(await missing.json()), /ENOENT|publication\.json|\/tmp\//);
    for (const query of ["?run=one&run=two", "?year=2024", "?run=..%2Fescape", "?run="]) {
      const response = await get(forecast, query);
      assert.equal(response.status, 400, query);
      assert.equal(response.headers.get("cache-control"), "no-store");
    }
    assert.equal((await get(availability, "?run=one")).status, 400);
    await mkdir(join(directory, "data/house-live"), { recursive: true });
    await writeFile(join(directory, "data/house-live/publication.json"), "{}");
    const broken = await get(availability);
    assert.equal(broken.status, 503);
    const body = await broken.json();
    assert.equal("available" in body, false);
    assert.doesNotMatch(JSON.stringify(body), /manifest|publication\.json|\/tmp\//);
    assert.equal((await get(forecast)).status, 503);
    assert.equal(serverErrors.length, 3);
  } finally {
    process.chdir(original);
    console.error = originalError;
    await rm(directory, { recursive: true, force: true });
  }
});
