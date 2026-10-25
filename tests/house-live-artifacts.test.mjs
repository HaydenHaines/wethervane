import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { readPinnedHouseLiveJson, verifyPinnedHouseLiveArtifact } from "../lib/house-live-artifacts.ts";

const digest = value => createHash("sha256").update(value).digest("hex");

async function fixture(t) {
  const base = await mkdtemp(join(tmpdir(), "ballotmancy-live-artifacts-"));
  t.after(() => rm(base, { recursive: true, force: true }));
  const directory = join(base, "publication");
  await mkdir(join(directory, "results"), { recursive: true });
  const bytes = Buffer.from('{"invented_test_fixture":true,"value":7}\n');
  await writeFile(join(directory, "results", "example.json"), bytes);
  return { base, directory, bytes, pin: { path: "results/example.json", sha256: digest(bytes) } };
}

test("reads authenticated nested JSON bytes without admitting them as a forecast", async t => {
  const { directory, pin } = await fixture(t);
  assert.deepEqual(await readPinnedHouseLiveJson(directory, pin), { invented_test_fixture: true, value: 7 });
  assert.equal(await verifyPinnedHouseLiveArtifact(directory, pin), undefined);
});

test("changed artifact bytes invalidate the reviewed digest", async t => {
  const { directory, pin } = await fixture(t);
  await writeFile(join(directory, pin.path), '{"invented_test_fixture":true,"value":8}\n');
  await assert.rejects(readPinnedHouseLiveJson(directory, pin), /fingerprint mismatch/);
  await assert.rejects(verifyPinnedHouseLiveArtifact(directory, pin), /fingerprint mismatch/);
});

test("rejects path traversal, absolute paths and ambiguous components", async t => {
  const { directory, pin } = await fixture(t);
  for (const path of ["../outside.json", "/tmp/outside.json", "results/../outside.json", "results//example.json", "results/./example.json", "results\\example.json", "https://example.com/file.json"]) {
    await assert.rejects(readPinnedHouseLiveJson(directory, { ...pin, path }), /Invalid live House artifact reference/);
  }
});

test("rejects symlink escapes even when the target has the expected digest", async t => {
  const { base, directory, bytes, pin } = await fixture(t);
  const outside = join(base, "outside.json");
  await writeFile(outside, bytes);
  await symlink(outside, join(directory, "escape.json"));
  await assert.rejects(readPinnedHouseLiveJson(directory, { ...pin, path: "escape.json" }), /inside the publication directory/);
  await assert.rejects(verifyPinnedHouseLiveArtifact(directory, { ...pin, path: "escape.json" }), /inside the publication directory/);
});

test("streams a binary artifact and detects a changed binary", async t => {
  const { directory } = await fixture(t);
  const bytes = Buffer.from([0, 1, 255, 2, 0, 0, 3, 128]);
  const pin = { path: "joint.bin", sha256: digest(bytes) };
  await writeFile(join(directory, pin.path), bytes);
  await verifyPinnedHouseLiveArtifact(directory, pin);
  await assert.rejects(readPinnedHouseLiveJson(directory, pin), /JSON filename/);
  await writeFile(join(directory, pin.path), Buffer.from([0, 1]));
  await assert.rejects(verifyPinnedHouseLiveArtifact(directory, pin), /fingerprint mismatch/);
});

test("a matching digest does not make malformed JSON or a directory an artifact", async t => {
  const { directory, pin } = await fixture(t);
  await assert.rejects(verifyPinnedHouseLiveArtifact(directory, { ...pin, path: "results" }), /must be a file/);
  const bytes = Buffer.from("{ not json }");
  await writeFile(join(directory, "invalid.json"), bytes);
  await assert.rejects(readPinnedHouseLiveJson(directory, { path: "invalid.json", sha256: digest(bytes) }), SyntaxError);
});
