import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile, chmod } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { makeHouseLiveResult } from "./fixtures/house-live-fixtures.mjs";
import { checkHouseLiveJointArchive } from "../lib/house-live-joint-archive.ts";

const options = { seatQuantileConvention: "inverse_cdf" };
function fourRows() {
  const rows = [Array(435).fill(0), Array(435).fill(0), Array(435).fill(1), Array(435).fill(1)];
  rows[1][1] = rows[3][1] = 2;
  rows[2][2] = rows[3][2] = 0;
  return rows;
}
async function fixture(t) {
  const directory = await mkdtemp(join(tmpdir(), "house-joint-bridge-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const result = makeHouseLiveResult();
  const path = "invented-joint.npz", rows = fourRows();
  const written = spawnSync("python3", [fileURLToPath(new URL("./fixtures/house-live-joint-archive.py", import.meta.url))],
    { input: JSON.stringify({ destination: join(directory, path), raceIds: result.joint_draws.ordered_race_ids,
      rows, chains: result.joint_draws.chain_count }), encoding: "utf8" });
  assert.equal(written.status, 0, written.stderr);
  const sha256 = createHash("sha256").update(await readFile(join(directory, path))).digest("hex");
  Object.assign(result.joint_draws, { path, sha256 });
  result.chamber.joint_draws_sha256 = sha256;
  return { directory, result, rows };
}

test("authenticated NPZ rows feed the existing whole-chamber numerical checker", async t => {
  const { directory, result } = await fixture(t);
  assert.deepEqual(await checkHouseLiveJointArchive(directory, result, options), {
    rowsChecked: 4, raceCount: 435, chamberChecked: true,
    artifactSha256: result.joint_draws.sha256, archiveFormat: "house_live_npz_chain_major_v1",
    seatQuantileConvention: "inverse_cdf",
  });
  await assert.rejects(checkHouseLiveJointArchive(directory, result, { seatQuantileConvention: "linear" }), /quantiles do not match/);
  await assert.rejects(checkHouseLiveJointArchive(directory, result, {}), /quantile convention/);
});

test("archive bytes and declared race identities must match before a check can succeed", async t => {
  const { directory, result } = await fixture(t);
  const badHash = structuredClone(result);
  badHash.joint_draws.sha256 = badHash.chamber.joint_draws_sha256 = "f".repeat(64);
  await assert.rejects(checkHouseLiveJointArchive(directory, badHash, options), /decoder failed/);
  const reordered = structuredClone(result);
  [reordered.races[0], reordered.races[3]] = [reordered.races[3], reordered.races[0]];
  reordered.joint_draws.ordered_race_ids = reordered.races.map(race => race.race_id);
  await assert.rejects(checkHouseLiveJointArchive(directory, reordered, options), /decoder failed/);
});

test("a helper that emits complete valid rows but exits unsuccessfully is never accepted", async t => {
  const { directory, result, rows } = await fixture(t);
  const executable = join(directory, "late-failure.py");
  await writeFile(executable, `#!/usr/bin/env python3\nimport sys\nsys.stdin.read()\nsys.stdout.buffer.write(bytes.fromhex("${Buffer.from(rows.flat()).toString("hex")}"))\nsys.stdout.buffer.flush()\nsys.stderr.write("Final archive verification failed")\nsys.exit(9)\n`);
  await chmod(executable, 0o700);
  await assert.rejects(checkHouseLiveJointArchive(directory, result, { ...options, pythonExecutable: executable }), /Final archive verification failed/);
});

test("missing interpreter and cancellation are explicit failures", async t => {
  const { directory, result } = await fixture(t);
  await assert.rejects(checkHouseLiveJointArchive(directory, result, {
    ...options, pythonExecutable: join(directory, "missing-python"),
  }), /decoder failed/);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(checkHouseLiveJointArchive(directory, result, { ...options, signal: controller.signal }), /abort/i);
});
