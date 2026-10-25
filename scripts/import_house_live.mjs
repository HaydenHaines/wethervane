/** Offline admission of an explicitly pinned, operator-reviewed native release.
 * Raw predictive archives are checked in place and never staged for the site.
 */
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { lstat, mkdir, mkdtemp, readFile, realpath, rm, stat, writeFile } from "node:fs/promises";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readPinnedHouseLiveJson, verifyPinnedHouseLiveArtifact } from "../lib/house-live-artifacts.ts";
import { parseHouseLiveCatalog } from "../lib/house-live-catalog.ts";
import { validateHouseLiveCollection } from "../lib/house-live-collection.ts";
import { checkHouseLiveJointArchive } from "../lib/house-live-joint-archive.ts";
import { parseHouseLivePublicationQualification } from "../lib/house-live-native-qualification.ts";
import { houseLiveConfigurationKey, parseHouseLiveResult } from "../lib/house-live-result.ts";
import { loadHouseLiveSitePublication, parseHouseLiveSiteReview, readHouseLiveAvailability } from "../lib/house-live-server.ts";

const require = (condition, message) => { if (!condition) throw Error(message); };
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const bytesOf = value => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
const inside = (root, path) => {
  const part = relative(root, path);
  return !part || (!isAbsolute(part) && part !== ".." && !part.startsWith(`..${sep}`));
};
const safePath = path => typeof path === "string" && /^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(path) &&
  path.split("/").every(part => part && part !== "." && part !== "..");
const plainText = value => typeof value === "string" && value.length > 0 && value.length <= 4000 &&
  value === value.trim() && !/[\u0000-\u001f\u007f]/.test(value);

/** Capture and hash the actual copied bytes, independently of any earlier read. */
async function readSafePinnedBytes(root, pin) {
  require(pin && safePath(pin.path) && /^[a-f0-9]{64}$/.test(pin.sha256), "Invalid live House staging pin.");
  const path = await realpath(resolve(root, pin.path));
  require(path !== root && inside(root, path) && (await stat(path)).isFile(),
    "Live House staging artifact must be a file inside its source root.");
  const bytes = await readFile(path);
  require(digest(bytes) === pin.sha256, "Live House staging artifact fingerprint mismatch.");
  return bytes;
}

async function assertAbsent(path) {
  try { await lstat(path); }
  catch (error) { if (error.code === "ENOENT") return; throw error; }
  throw Error("Live House output destination already exists; choose a new explicit directory.");
}

async function installNoReplace(parent, staging, output, pythonExecutable) {
  const helper = fileURLToPath(new URL("./install_house_live_release.py", import.meta.url));
  const child = spawn(pythonExecutable ?? "python3", [helper, "--parent", parent,
    "--source", basename(staging), "--destination", basename(output)], { stdio: ["ignore", "ignore", "pipe"] });
  let stderr = "";
  child.stderr.on("data", chunk => { stderr = (stderr + chunk.toString("utf8")).slice(-8192); });
  await new Promise((resolve, reject) => {
    child.once("error", error => reject(Error(`Live House release installer failed: ${error.message}`)));
    child.once("close", (code, signal) => code === 0 ? resolve() :
      reject(Error(`Live House release installer failed: ${stderr.trim() || signal || `exit ${code}`}`)));
  });
}

export async function importHouseLive({ sourceRoot, catalog: catalogPin, reviewRoot, review: reviewPin,
  geographySource, outputDirectory, pythonExecutable } = {}) {
  require([sourceRoot, reviewRoot, outputDirectory].every(value => typeof value === "string" && value.length > 0),
    "Explicit source root, review root, and output directory are required.");
  require(geographySource && plainText(geographySource.name) && plainText(geographySource.url) &&
    /^https?:\/\//i.test(geographySource.url) &&
    (geographySource.note === undefined || plainText(geographySource.note)), "Explicit geography source attribution is required.");
  require(pythonExecutable === undefined || (typeof pythonExecutable === "string" && pythonExecutable.length > 0),
    "Invalid offline Python executable.");
  const root = await realpath(sourceRoot), operatorRoot = await realpath(reviewRoot);
  const output = resolve(outputDirectory), parent = dirname(output);
  require((await realpath(parent)) === parent, "Live House output parent must not traverse symlinks.");
  require(!inside(root, output) && !inside(operatorRoot, output), "Live House output must be outside both read-only source roots.");
  await assertAbsent(output);

  const files = new Map();
  let reservedReviewPath;
  function stage(path, bytes) {
    require(safePath(path) && path !== "publication.json" && !path.startsWith("publication.json/"),
      "Staged native artifact conflicts with the publication manifest.");
    require(!/\.npz$/i.test(path), "Raw NPZ archives must never be staged for the site.");
    require(path !== reservedReviewPath, "Native artifact conflicts with the operator review path.");
    const existing = files.get(path);
    require(!existing || existing.equals(bytes), "Conflicting staged artifact paths.");
    for (const other of files.keys()) require(!path.startsWith(`${other}/`) && !other.startsWith(`${path}/`),
      "Staged artifact file and parent paths conflict.");
    files.set(path, bytes);
  }
  async function json(source, pin) {
    const value = await readPinnedHouseLiveJson(source, pin);
    return { value, bytes: await readSafePinnedBytes(source, pin) };
  }
  async function opaque(pin) {
    await verifyPinnedHouseLiveArtifact(root, pin);
    stage(pin.path, await readSafePinnedBytes(root, pin));
  }

  const catalogFile = await json(root, catalogPin);
  const catalog = parseHouseLiveCatalog(catalogFile.value);
  require(catalog, "Invalid native live House catalog.");
  const reviewFile = await json(operatorRoot, reviewPin);
  const review = parseHouseLiveSiteReview(reviewFile.value);
  require(review.catalog_sha256 === catalogPin.sha256 &&
    review.source_policy_sha256 === catalog.source_policy.sha256 &&
    review.qualification_sha256 === catalog.qualification.sha256,
  "Operator review does not bind the native catalog, source policy, and qualification.");
  stage(catalogPin.path, catalogFile.bytes);
  // The review is operator-owned and can originate in an entirely separate tree.
  const stagedReview = { path: `operator-review-${reviewPin.sha256}.json`, sha256: reviewPin.sha256 };
  require(!files.has(stagedReview.path), "Native artifact conflicts with the operator review path.");
  stage(stagedReview.path, reviewFile.bytes);
  reservedReviewPath = stagedReview.path;
  await opaque(catalog.source_policy);
  const qualificationFile = await json(root, catalog.qualification);
  const qualification = parseHouseLivePublicationQualification(qualificationFile.value);
  require(qualification && qualification.seat_quantile_convention === review.seat_quantile_convention,
    "Native House qualification is unreviewed or disagrees with the reviewed export convention.");
  stage(catalog.qualification.path, qualificationFile.bytes);
  await opaque(qualification.numerical_review);

  const reviewed = new Map(review.results.map(row => [row.id, row]));
  const results = [];
  for (const entry of catalog.entries) {
    const file = await json(root, entry.result);
    const result = parseHouseLiveResult(file.value), decision = reviewed.get(entry.id);
    require(result && decision && result.id === entry.id &&
      houseLiveConfigurationKey(result.configuration) === houseLiveConfigurationKey(entry.configuration) &&
      decision.result_sha256 === entry.result.sha256 && decision.joint_draws_sha256 === result.joint_draws.sha256 &&
      result.qualification.publication_status === "reviewed" &&
      qualification.limitations.every(note => result.qualification.limitations.includes(note)),
    `Native result ${entry.id} does not match its reviewed identity and qualification.`);
    require(result.geography.sha256 === review.geography_sha256 &&
      result.geography.source_receipt.sha256 === review.geography_source_receipt_sha256,
    `Native result ${entry.id} does not match the reviewed geography.`);
    require(entry.result.path !== stagedReview.path, "Native artifact conflicts with the operator review path.");
    stage(entry.result.path, file.bytes);
    results.push(result);
  }
  const collection = validateHouseLiveCollection(catalog, results);
  require(collection.valid, `Invalid live House collection: ${collection.issues.join(" ")}`);
  const geography = results[0].geography;
  const geographyFile = await json(root, geography);
  require(geography.path !== stagedReview.path && geography.source_receipt.path !== stagedReview.path,
    "Native artifact conflicts with the operator review path.");
  stage(geography.path, geographyFile.bytes);
  await opaque(geography.source_receipt);
  for (const result of results) require(!files.has(result.joint_draws.path),
    "A staged artifact path conflicts with a raw joint archive.");

  let jointRowsChecked = 0;
  // Check each native result independently, even when multiple runs share an archive.
  for (const result of results) {
    const checked = await checkHouseLiveJointArchive(root, result, {
      seatQuantileConvention: review.seat_quantile_convention,
      ...(pythonExecutable === undefined ? {} : { pythonExecutable }),
    });
    jointRowsChecked += checked.rowsChecked;
  }
  const manifest = {
    schema: "HouseLiveSiteManifestV1", status: "available", cycle: 2026,
    catalog: { path: catalogPin.path, sha256: catalogPin.sha256 }, review: stagedReview,
    sourcePolicy: { path: catalog.source_policy.path, sha256: catalog.source_policy.sha256 },
    qualification: { path: catalog.qualification.path, sha256: catalog.qualification.sha256 },
    geography: { artifact: { path: geography.path, sha256: geography.sha256 },
      sourceReceipt: { path: geography.source_receipt.path, sha256: geography.source_receipt.sha256 },
      sourceName: geographySource.name, sourceUrl: geographySource.url,
      ...(geographySource.note === undefined ? {} : { sourceNote: geographySource.note }) },
    results: catalog.entries.map(entry => ({ id: entry.id,
      artifact: { path: entry.result.path, sha256: entry.result.sha256 } })),
  };
  const staging = await mkdtemp(join(parent, ".house-live-import-"));
  try {
    for (const [path, bytes] of files) {
      await mkdir(dirname(join(staging, path)), { recursive: true });
      await writeFile(join(staging, path), bytes, { flag: "wx" });
    }
    await writeFile(join(staging, "publication.json"), bytesOf(manifest), { flag: "wx" });
    for (const entry of catalog.entries) await loadHouseLiveSitePublication(staging, entry.id);
    require((await readHouseLiveAvailability(staging)).available, "Staged live House release is unavailable.");
    require((await realpath(parent)) === parent, "Live House output parent changed or traverses symlinks.");
    await installNoReplace(parent, staging, output, pythonExecutable);
    return { outputDirectory: output, results: results.length, jointRowsChecked,
      catalogSha256: catalogPin.sha256, reviewSha256: reviewPin.sha256 };
  } finally {
    await rm(staging, { recursive: true, force: true });
  }
}

async function main() {
  const options = {}, names = {
    "--source-root": "sourceRoot", "--catalog": "catalogPath", "--catalog-sha256": "catalogSha256",
    "--review-root": "reviewRoot", "--review": "reviewPath", "--review-sha256": "reviewSha256",
    "--geography-source-name": "geographyName", "--geography-source-url": "geographyUrl",
    "--geography-source-note": "geographyNote", "--output-dir": "outputDirectory", "--python": "pythonExecutable",
  };
  const args = process.argv.slice(2);
  for (let index = 0; index < args.length; index += 2) {
    const name = names[args[index]];
    require(name && args[index + 1] && options[name] === undefined, "Expected explicit, nonrepeated offline importer options.");
    options[name] = args[index + 1];
  }
  console.log(JSON.stringify(await importHouseLive({ ...options,
    catalog: { path: options.catalogPath, sha256: options.catalogSha256 },
    review: { path: options.reviewPath, sha256: options.reviewSha256 },
    geographySource: { name: options.geographyName, url: options.geographyUrl,
      ...(options.geographyNote === undefined ? {} : { note: options.geographyNote }) },
  })));
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
