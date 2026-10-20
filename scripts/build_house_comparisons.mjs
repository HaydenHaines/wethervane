#!/usr/bin/env node
/** Build a reviewed comparison allowlist; no fitting, draw pairing or new predictions. */
import { createHash } from "node:crypto";
import { readFile, writeFile, realpath } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  HOUSE_COMPARISON_SOURCE_PINS,
  houseComparisonCanonicalJson,
  houseComparisonIdentity,
  houseComparisonSupport,
  parseHouseComparisonManifest,
} from "../lib/house-comparison.ts";
import { parseHouseResult } from "../lib/house-result.ts";

const RESEARCH = "research/house-benchmark-20261020";
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const canonicalHash = (value) => hash(houseComparisonCanonicalJson(value));
const read = async (file) => JSON.parse(await readFile(file, "utf8"));
const require = (condition, message) => {
  if (!condition) throw Error(message);
};
async function pinned(root, reference) {
  require(reference &&
    typeof reference.path === "string" &&
    !path.isAbsolute(reference.path) &&
    !reference.path.split(/[\\/]/).includes(".."), "Unsafe source reference.");
  const rootPath = await realpath(root),
    file = await realpath(path.join(root, reference.path));
  require(file.startsWith(
    rootPath + path.sep,
  ), "Source reference leaves the source checkout.");
  require(hash(await readFile(file)) ===
    reference.sha256, "A reviewed source fingerprint changed.");
  return file;
}

export async function buildHouseComparisonManifest({
  sourceRoot,
  publicationDirectory,
}) {
  const files = {
    sourceCatalog: `${RESEARCH}/frontend-results-v2/catalog.json`,
    reviewReceipt: `${RESEARCH}/frontend-results-v2/review-receipt.json`,
    sourceCapabilityPolicy: `${RESEARCH}/source-capability-policy-v1.json`,
    sourceHandoff: `${RESEARCH}/HANDOFF.md`,
    sourcePanel: `${RESEARCH}/panel-v6/receipt.json`,
    interpretationReceipt: `${RESEARCH}/interpretation-v1/receipt.json`,
  };
  const verified = {};
  for (const [key, file] of Object.entries(files))
    verified[key] = await pinned(sourceRoot, {
      path: file,
      sha256: HOUSE_COMPARISON_SOURCE_PINS[key],
    });
  const [sourceCatalog, review, policy] = await Promise.all([
    read(verified.sourceCatalog),
    read(verified.reviewReceipt),
    read(verified.sourceCapabilityPolicy),
  ]);
  require(policy.target_id ===
    "house_aggregate_fundamentals_retrospective_v1" &&
    policy.nominal_horizon_days === 14 &&
    policy.denominator === "Democratic and Republican candidate votes" &&
    policy.information_mode ===
      "retrospective_proxy_strictly_earlier_fitted_labels", "Reviewed target, horizon or denominator changed.");
  require(policy.capabilities.historical_races === true &&
    [
      "polling",
      "district_geometry",
      "full_chamber",
      "chamber_control",
      "chamber_tipping",
      "current_2026",
    ].every(
      (key) => policy.capabilities[key] === false,
    ), "Comparison capability scope changed.");
  require(sourceCatalog.configurations.length === 16 &&
    review.cells.length ===
      16, "Expected sixteen reviewed model/year results.");
  const publication = await read(
    path.join(publicationDirectory, "publication.json"),
  );
  require(publication.status === "available" &&
    publication.catalog.sha256 ===
      HOUSE_COMPARISON_SOURCE_PINS.publishedCatalog &&
    path.basename(publication.catalog.file) ===
      publication.catalog.file, "Published House catalog changed.");
  const catalogBytes = await readFile(
    path.join(publicationDirectory, publication.catalog.file),
  );
  require(hash(catalogBytes) ===
    publication.catalog.sha256, "Published catalog fingerprint mismatch.");
  const catalog = JSON.parse(catalogBytes);
  require(catalog.runs.length === 16, "Published collection is incomplete.");
  const groups = new Map();
  for (const entry of sourceCatalog.configurations) {
    require(entry.publication_eligible ===
      true, "An unqualified result cannot be admitted for comparison.");
    const sourcePath = await pinned(sourceRoot, entry.result),
      sourceBytes = await readFile(sourcePath);
    const exposed = catalog.runs.find((run) => run.id === entry.id);
    require(exposed?.available &&
      exposed.resultSha256 === entry.result.sha256 &&
      exposed.result.sha256 === entry.result.sha256 &&
      path.basename(exposed.result.file) ===
        exposed.result.file, "Published result binding changed.");
    const bytes = await readFile(
      path.join(publicationDirectory, exposed.result.file),
    );
    require(hash(bytes) === exposed.resultSha256 &&
      bytes.equals(
        sourceBytes,
      ), "Published result differs from its reviewed source.");
    const result = parseHouseResult(JSON.parse(bytes));
    require(result, "A reviewed result failed the frozen parser.");
    require(result.run.configurationId === entry.configurationId &&
      result.run.configurationSha256 === entry.configurationSha256 &&
      result.run.year === entry.year &&
      result.run.informationMode === policy.information_mode &&
      result.metadata.full_chamber === false &&
      result.metadata.scope ===
        "eligible_race_subset", "Result scientific context changed.");
    const cell = review.cells.find(
      (cell) =>
        cell.cycle === entry.year &&
        cell.configuration === entry.configurationId,
    );
    require(cell?.numerical_status === "passed" &&
      cell.structural_and_score_audit_passed === true &&
      cell.subset_seat_precision_passed ===
        false, "Race qualification or subtotal restriction changed.");
    const receiptPath = await pinned(sourceRoot, cell.fit_receipt),
      receipt = await read(receiptPath);
    const planPath = await pinned(sourceRoot, receipt.plan),
      plan = await read(planPath);
    require(receipt.status === "passed" &&
      receipt.race_prediction_precision_passed === true &&
      receipt.subset_seat_precision_passed ===
        false, "Fit/race qualification changed.");
    require(receipt.plan.sha256 === result.run.configurationSha256 &&
      plan.house_configuration_id === result.run.configurationId &&
      plan.target_cycle === entry.year &&
      plan.target_id === policy.target_id &&
      plan.forecast_horizon_days === policy.nominal_horizon_days &&
      plan.information_mode === policy.information_mode &&
      plan.scope === "eligible_race_subset" &&
      plan.full_chamber === false &&
      plan.panel.sha256 ===
        HOUSE_COMPARISON_SOURCE_PINS.sourcePanel, "Plan does not admit this exact same-year comparison.");
    await pinned(sourceRoot, plan.panel);
    for (const pin of plan.source_pins) await pinned(sourceRoot, pin);
    const raceIds = result.races.map((race) => race.id).sort();
    require(JSON.stringify([...plan.target_race_ids].sort()) ===
      JSON.stringify(raceIds) &&
      result.races.length === policy.coverage[String(entry.year)] &&
      result.races.every(
        (race) =>
          race.observed &&
          race.outcomes.every(
            (outcome) => outcome.share?.denominator === policy.denominator,
          ),
      ), "Source-plan race support or observed truth is incomplete.");
    const predictions = receipt.outputs.find(
      (output) => path.basename(output.path) === "predictions.npz",
    );
    require(result.jointDraws &&
      predictions?.sha256 === result.jointDraws.sourceSha256 &&
      result.jointDraws.configurationId === result.run.configurationId &&
      result.jointDraws.cycle ===
        entry.year, "Joint reference does not match the reviewed fit receipt.");
    const supportSha256 = canonicalHash(houseComparisonSupport(result)),
      identitySha256 = canonicalHash(houseComparisonIdentity(result));
    let group = groups.get(entry.year);
    if (!group) {
      group = {
        year: entry.year,
        mapVintage: result.metadata.mapVintage,
        raceCount: raceIds.length,
        raceIds,
        supportSha256,
        identitySha256,
        runs: [],
      };
      groups.set(entry.year, group);
    }
    require(group.supportSha256 === supportSha256 &&
      group.identitySha256 ===
        identitySha256, "Same-year models have different support, candidates or actual outcomes.");
    group.runs.push({
      runId: result.run.id,
      configurationId: result.run.configurationId,
      configurationSha256: result.run.configurationSha256,
      resultSha256: exposed.resultSha256,
      canonicalResultSha256: canonicalHash(result),
      planSha256: receipt.plan.sha256,
      fitReceiptSha256: cell.fit_receipt.sha256,
      jointArtifactId: result.jointDraws.id,
      jointSourceSha256: result.jointDraws.sourceSha256,
    });
  }
  const manifest = {
    schemaVersion: 1,
    id: "reviewed-house-same-year-20261020-v1",
    targetId: policy.target_id,
    denominator: policy.denominator,
    nominalHorizonDays: policy.nominal_horizon_days,
    informationMode: policy.information_mode,
    scope: "eligible_race_subset",
    sourcePins: { ...HOUSE_COMPARISON_SOURCE_PINS },
    groups: [...groups.values()]
      .sort((a, b) => a.year - b.year)
      .map((group) => ({
        ...group,
        runs: group.runs.sort((a, b) => a.runId.localeCompare(b.runId)),
      })),
    limitations: [
      "Historical comparison of the same eligible races in the same election and district map edition; retrospective source qualifications remain.",
      "Each model uses its own independently seeded joint artifact. Simulation rows must not be paired, blended or treated as counterfactual differences.",
      "Only categorical win scores and changes in saved race summaries are supported. Public results do not supply actual vote shares or full predictive distributions for share CRPS or interval scoring.",
      "All sixteen eligible-seat-subtotal precision gates failed. No full-chamber, control, tipping, current forecast or geometry claim is admitted.",
      "Numerical differences are descriptive; no significance test, causal attribution or automatic change of the Fixed reference is implied.",
      "Context's 2018 comparison extrapolates beyond its training presidential-party context; its priors materially affect that result.",
    ],
  };
  require(parseHouseComparisonManifest(
    manifest,
  ), "Generated comparison manifest failed validation.");
  return manifest;
}

async function main() {
  const args = process.argv.slice(2),
    options = {};
  for (let i = 0; i < args.length; i += 2) {
    require(["--source-root", "--publication-dir", "--output"].includes(
      args[i],
    ) &&
      args[
        i + 1
      ], "Expected --source-root, optional --publication-dir and --output.");
    options[args[i]] = args[i + 1];
  }
  require(options["--source-root"], "A read-only source root is required.");
  const manifest = await buildHouseComparisonManifest({
    sourceRoot: options["--source-root"],
    publicationDirectory: options["--publication-dir"] ?? "data/house",
  });
  await writeFile(
    options["--output"] ?? "data/house-comparisons.json",
    JSON.stringify(manifest, null, 2) + "\n",
  );
  console.log(
    `Bound ${manifest.groups.reduce((sum, group) => sum + group.runs.length, 0)} reviewed House results in ${manifest.groups.length} same-year groups; no model execution.`,
  );
}
if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  await main();
