import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { loadHousePublication } from "../lib/house-publication.ts";
import { houseReadinessInput, HOUSE_SHARE_DENOMINATOR } from "../lib/house-result.ts";
import { makeHousePublication } from "./fixtures/house.mjs";

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const jsonBytes = (value) => Buffer.from(JSON.stringify(value));

async function inPublicationDirectory(fn) {
  const directory = await mkdtemp(join(tmpdir(), "house-publication-"));
  try {
    await fn(directory);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

async function writePublication(directory, publication, patchManifest = (value) => value) {
  const resultBytes = jsonBytes(publication.result);
  await writeFile(join(directory, "result.json"), resultBytes);
  const manifest = {
    schemaVersion: 1,
    status: "available",
    result: { file: "result.json", sha256: sha256(resultBytes) },
  };
  if (publication.geography) {
    const geographyBytes = jsonBytes(publication.geography.geojson);
    await writeFile(join(directory, "geography.json"), geographyBytes);
    manifest.geography = {
      file: "geography.json",
      sha256: sha256(geographyBytes),
      metadata: publication.geography.metadata,
    };
  }
  await writeFile(join(directory, "publication.json"), jsonBytes(patchManifest(manifest)));
  return manifest;
}

test("preparing publication is a safe, data-free state", async () => {
  await inPublicationDirectory(async (directory) => {
    await writeFile(join(directory, "publication.json"), JSON.stringify({
      schemaVersion: 1,
      status: "preparing",
      result: { file: "../must-not-be-read.json", sha256: "bad" },
    }));
    assert.deepEqual(await loadHousePublication(directory), { schemaVersion: 1, status: "preparing" });
  });
});

test("reviewed subset with hash-pinned geometry is admitted", async () => {
  await inPublicationDirectory(async (directory) => {
    const publication = makeHousePublication({ eligibleCount: 16, year: 2022, withGeography: true });
    const manifest = await writePublication(directory, publication);
    const loaded = await loadHousePublication(directory);
    assert.equal(loaded.status, "available");
    if (loaded.status !== "available") return;
    assert.equal(loaded.result.races.length, 16);
    assert.equal(loaded.resultSha256, manifest.result.sha256);
    assert.equal(loaded.geography?.metadata.sourceSha256, manifest.geography.sha256);
  });
});

test("draft results cannot enter the publication path", async () => {
  await inPublicationDirectory(async (directory) => {
    const publication = makeHousePublication({ eligibleCount: 16, withGeography: false });
    publication.result.reviewStatus = "draft";
    await writePublication(directory, publication);
    await assert.rejects(loadHousePublication(directory), /publication checks/);
  });
});

test("configuration identity must agree between run and forecast metadata", async () => {
  await inPublicationDirectory(async (directory) => {
    const publication = makeHousePublication({ eligibleCount: 16, withGeography: false });
    publication.result.metadata.configurationId = "different-configuration";
    await writePublication(directory, publication);
    await assert.rejects(loadHousePublication(directory), /publication checks/);
  });
});

test("artifact bytes must match the manifest fingerprint", async () => {
  await inPublicationDirectory(async (directory) => {
    const publication = makeHousePublication({ eligibleCount: 16, withGeography: false });
    await writePublication(directory, publication, (manifest) => {
      manifest.result.sha256 = "0".repeat(64);
      return manifest;
    });
    await assert.rejects(loadHousePublication(directory), /fingerprint mismatch/);
  });
});

test("artifact references cannot traverse outside the publication directory", async () => {
  await inPublicationDirectory(async (directory) => {
    const publication = makeHousePublication({ eligibleCount: 16, withGeography: false });
    await writePublication(directory, publication, (manifest) => {
      manifest.result.file = "../outside.json";
      return manifest;
    });
    await assert.rejects(loadHousePublication(directory), /Invalid House artifact reference/);
  });
});

test("artifact symlinks cannot make the loader read outside the publication directory", async () => {
  await inPublicationDirectory(async (directory) => {
    const publication = makeHousePublication({ eligibleCount: 16, withGeography: false });
    const outside = join(directory, "..", "house-publication-outside.json");
    const bytes = jsonBytes(publication.result);
    try {
      await writeFile(outside, bytes);
      await symlink(outside, join(directory, "result.json"));
    } catch (error) {
      await rm(outside, { force: true });
      if (["EPERM", "EACCES", "ENOTSUP"].includes(error.code)) return;
      throw error;
    }
    await writeFile(join(directory, "publication.json"), jsonBytes({
      schemaVersion: 1,
      status: "available",
      result: { file: "result.json", sha256: sha256(bytes) },
    }));
    try {
      await assert.rejects(loadHousePublication(directory));
    } finally {
      await rm(outside, { force: true });
    }
  });
});

test("manifest symlinks cannot make the loader read outside the publication directory", async () => {
  await inPublicationDirectory(async (directory) => {
    const outside = join(directory, "..", "house-publication-manifest.json");
    await writeFile(outside, JSON.stringify({ schemaVersion: 1, status: "preparing" }));
    try {
      await symlink(outside, join(directory, "publication.json"));
    } catch (error) {
      await rm(outside, { force: true });
      if (["EPERM", "EACCES", "ENOTSUP"].includes(error.code)) return;
      throw error;
    }
    try {
      await assert.rejects(loadHousePublication(directory));
    } finally {
      await rm(outside, { force: true });
    }
  });
});

test("coverage ledger must contain all 435 unique districts and every eligible race", async (t) => {
  const cases = [
    ["missing coverage", (result) => { result.coverage.pop(); }],
    ["eligible race absent", (result) => {
      const row = result.coverage.find((entry) => entry.eligible);
      result.races = result.races.filter((race) => race.id !== row.raceId);
    }],
    ["excluded race included", (result) => {
      const row = result.coverage.find((entry) => !entry.eligible);
      const source = result.coverage.find((entry) => entry.eligible);
      const race = result.races.find((entry) => entry.id === source.raceId);
      result.coverage[result.coverage.indexOf(row)] = { ...row, eligible: true, exclusionReason: null };
      result.races.push({ ...race, id: row.raceId, district: { ...race.district, districtKey: row.districtKey } });
    }],
    ["duplicate district key", (result) => {
      result.coverage[1].districtKey = result.coverage[0].districtKey;
    }],
  ];
  for (const [name, mutate] of cases) {
    await t.test(name, async () => inPublicationDirectory(async (directory) => {
      const publication = makeHousePublication({ eligibleCount: 16, withGeography: false });
      mutate(publication.result);
      await writePublication(directory, publication);
      await assert.rejects(loadHousePublication(directory), /publication checks/);
    }));
  }
});

test("race outcomes require valid probabilities and observed winner identity", async (t) => {
  for (const [name, mutate] of [
    ["probability outside bounds", (race) => { race.outcomes[0].winProbability = 1.1; }],
    ["observed winner conflicts with outcome", (race) => {
      race.observed.outcomeId = race.outcomes.find((outcome) => outcome.party !== race.observed.winnerParty).id;
    }],
    ["observed winner name conflicts with outcome", (race) => {
      race.observed.winnerName = "A different person";
    }],
  ]) {
    await t.test(name, async () => inPublicationDirectory(async (directory) => {
      const publication = makeHousePublication({ eligibleCount: 16, withGeography: false });
      mutate(publication.result.races[0]);
      await writePublication(directory, publication);
      await assert.rejects(loadHousePublication(directory), /publication checks/);
    }));
  }
});

test("two-candidate share fields use the published denominator and complementary distributions", async (t) => {
  assert.equal(HOUSE_SHARE_DENOMINATOR, "Democratic and Republican candidate votes");
  const withComplementaryShares = (result) => {
    const [democrat, republican] = result.races[0].outcomes;
    democrat.share = {
      mean: 0.6,
      denominator: HOUSE_SHARE_DENOMINATOR,
      quantiles: { p05: 0.5, p50: 0.6, p95: 0.7 },
    };
    republican.share = {
      mean: 0.4,
      denominator: HOUSE_SHARE_DENOMINATOR,
      quantiles: { p05: 0.3, p50: 0.4, p95: 0.5 },
    };
  };
  for (const [name, mutate] of [
    ["noncanonical denominator", (democrat) => { democrat.share.denominator = "Two-party share"; }],
    ["noncomplementary means", (_democrat, republican) => { republican.share.mean = 0.45; }],
    ["noncomplementary quantiles", (_democrat, republican) => { republican.share.quantiles.p95 = 0.55; }],
  ]) {
    await t.test(name, async () => inPublicationDirectory(async (directory) => {
      const publication = makeHousePublication({ eligibleCount: 16, withGeography: false });
      withComplementaryShares(publication.result);
      mutate(...publication.result.races[0].outcomes);
      await writePublication(directory, publication);
      await assert.rejects(loadHousePublication(directory), /publication checks/);
    }));
  }
});

test("readiness input withholds tipping methods with unrecognized semantics", () => {
  const result = makeHousePublication({ eligibleCount: 16, withGeography: false }).result;
  result.tippingProof = {
    method: "some-other-ranking-method",
    sourceSha256: "a".repeat(64),
    jointDrawsId: "synthetic-joint",
    jointArtifactSha256: "b".repeat(64),
    configurationId: result.run.configurationId,
    cycle: result.run.year,
    scenarios: [],
  };
  assert.equal(houseReadinessInput(result, null).tippingProof, undefined);
});

test("geometry must match the cycle and map vintage and be explicitly verified", async (t) => {
  for (const [name, patch] of [
    ["wrong map vintage", { mapVintage: "unrelated-map-edition" }],
    ["unverified source", { admission: "unverified" }],
  ]) {
    await t.test(name, async () => inPublicationDirectory(async (directory) => {
      const publication = makeHousePublication({ eligibleCount: 16, withGeography: true });
      await writePublication(directory, publication, (manifest) => {
        Object.assign(manifest.geography.metadata, patch);
        return manifest;
      });
      await assert.rejects(loadHousePublication(directory), /geography has not passed publication checks/);
    }));
  }
});
