/** Verify one native candidate offline. This never stages or admits a publication. */
import { realpath } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { readPinnedHouseLiveJson } from "../lib/house-live-artifacts.ts";
import { checkHouseLiveJointArchive } from "../lib/house-live-joint-archive.ts";
import { parseHouseLiveResult } from "../lib/house-live-result.ts";
import { readHouseLivePredictionEvidence, resolveHouseLivePredictionPlan } from "./house_live_evidence.mjs";

const require = (condition, message) => { if (!condition) throw Error(message); };

export async function checkHouseLiveCandidate({ sourceRoot, result: resultPin, seatQuantileConvention,
  pythonExecutable } = {}) {
  require(typeof sourceRoot === "string" && sourceRoot.length > 0, "An explicit candidate source root is required.");
  require(["inverse_cdf", "linear"].includes(seatQuantileConvention),
    "An explicit candidate seat-quantile convention is required.");
  require(pythonExecutable === undefined || (typeof pythonExecutable === "string" && pythonExecutable.length > 0),
    "Invalid offline Python executable.");
  const root = await realpath(sourceRoot);
  const result = parseHouseLiveResult(await readPinnedHouseLiveJson(root, resultPin));
  require(result, "Invalid native House candidate result.");
  const evidence = await readHouseLivePredictionEvidence(root, result);
  const binding = resolveHouseLivePredictionPlan(result, evidence);
  require(binding, "Candidate verification requires an authenticated local prediction receipt and plan pair.");
  const jointArchive = await checkHouseLiveJointArchive(root, result, { seatQuantileConvention,
    ...(binding.kind === "combined" ? { combinedPlan: binding.plan } : {}),
    ...(pythonExecutable === undefined ? {} : { pythonExecutable }) });
  return { schema: "HouseLiveCandidateVerificationV1", publicationAdmission: false,
    notice: "Offline joint-row consistency verification only; this does not admit the candidate for publication.",
    result: { id: result.id, sha256: resultPin.sha256 },
    qualification: structuredClone(result.qualification),
    predictionEvidence: { kind: binding.kind, planSha256: binding.plan.sha256, receiptSha256: binding.receipt.sha256 },
    jointArchive };
}

async function main() {
  const options = {}, names = { "--source-root": "sourceRoot", "--result": "resultPath",
    "--result-sha256": "resultSha256", "--seat-quantile-convention": "seatQuantileConvention", "--python": "pythonExecutable" };
  const args = process.argv.slice(2);
  for (let index = 0; index < args.length; index += 2) {
    const name = names[args[index]];
    require(name && args[index + 1] && options[name] === undefined, "Expected explicit, nonrepeated candidate verification options.");
    options[name] = args[index + 1];
  }
  console.log(JSON.stringify(await checkHouseLiveCandidate({ ...options,
    result: { path: options.resultPath, sha256: options.resultSha256 } })));
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
