#!/usr/bin/env node
/** Export reviewed saved share-test summaries only. Never reads posterior arrays. */
import { createHash } from "node:crypto";
import { readFile, writeFile, realpath } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildHouseComparisonManifest } from "./build_house_comparisons.mjs";
import { houseComparisonCanonicalJson } from "../lib/house-comparison.ts";
import {
  HOUSE_SCORE_EVIDENCE_SOURCE_PINS,
  parseHouseScoreEvidenceManifest,
} from "../lib/house-score-evidence.ts";
const BASE = "research/house-benchmark-20261020";
const digest = (v) => createHash("sha256").update(v).digest("hex");
const check = (ok, message) => {
  if (!ok) throw Error(message);
};
const close = (a, b) =>
  Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) < 1e-12;
export function parseCsv(text) {
  const rows = [];
  let row = [],
    cell = "",
    quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else quoted = !quoted;
    } else if (!quoted && (c === "," || c === "\n")) {
      row.push(cell.replace(/\r$/, ""));
      cell = "";
      if (c === "\n") {
        rows.push(row);
        row = [];
      }
    } else cell += c;
  }
  check(!quoted, "Unterminated CSV quote.");
  if (cell || row.length) {
    row.push(cell.replace(/\r$/, ""));
    rows.push(row);
  }
  const headers = rows.shift();
  check(
    headers && new Set(headers).size === headers.length,
    "Invalid CSV headers.",
  );
  return rows
    .filter((r) => r.some(Boolean))
    .map((r) => {
      check(r.length === headers.length, "CSV row width differs.");
      return Object.fromEntries(headers.map((h, i) => [h, r[i]]));
    });
}
const uniqueIndex = (rows, key) => {
  const m = new Map();
  for (const row of rows) {
    check(!m.has(row[key]), `Duplicate ${key}.`);
    m.set(row[key], row);
  }
  return m;
};
/** Independently replay the saved score definitions against reviewed public predictions. */
export function validateHouseScoreRows(result, rows, labels, features) {
  const by = uniqueIndex(rows, "race_id"),
    truth = uniqueIndex(labels, "race_id"),
    roster = uniqueIndex(features, "race_id");
  check(
    by.size === result.races.length && result.races.every((r) => by.has(r.id)),
    "Incomplete score race support.",
  );
  const totals = {
    total: rows.length,
    covered: 0,
    meanAbsoluteError: 0,
    meanIntervalWidth: 0,
    crps: 0,
    intervalScore: 0,
  };
  for (const race of result.races) {
    const s = by.get(race.id),
      label = truth.get(race.id),
      f = roster.get(race.id);
    check(label && f, "Missing pinned truth or roster join.");
    check(
      s.configuration === result.run.configurationId &&
        +s.cycle === result.run.year &&
        s.numerical_status === "passed" &&
        +label.cycle === result.run.year &&
        +f.cycle === result.run.year &&
        f.map_vintage === result.metadata.mapVintage,
      "Score context differs.",
    );
    const d = race.outcomes.find((o) => o.party === "D"),
      r = race.outcomes.find((o) => o.party === "R");
    check(d && r && race.outcomes.length === 2, "Unsupported candidate pair.");
    for (const [outcome, prefix] of [
      [d, "dem"],
      [r, "rep"],
    ]) {
      const name =
        label[`official_${prefix}_candidate`] || f[`${prefix}_candidate`];
      const id = `${result.run.year}:${f.district_key}:${outcome.party}:${digest(name).slice(0, 16)}`;
      check(
        outcome.name === name &&
          outcome.id === id &&
          outcome.members.length === 1 &&
          outcome.members[0].id === id,
        "Candidate identity differs.",
      );
    }
    const actual = +s.actual_dem_share,
      mean = +s.mean,
      lower = +s.lower,
      upper = +s.upper,
      ae = +s.absolute_error,
      width = +s.interval_width,
      crps = +s.crps,
      intervalScore = +s.interval_score;
    check(
      [actual, mean, lower, upper].every(
        (v) => Number.isFinite(v) && v >= 0 && v <= 1,
      ) &&
        actual > 0 &&
        actual < 1 &&
        lower <= upper &&
        actual !== 0.5,
      "Invalid conditional D/R share or unsupported observed tie.",
    );
    check(
      close(actual, +label.actual_dem_share) &&
        close(actual, +label.dem_votes / (+label.dem_votes + +label.rep_votes)),
      "Observed vote denominator differs.",
    );
    check(
      race.observed?.outcomeId === (actual > 0.5 ? d.id : r.id) &&
        close(d.winProbability, +s.dem_win_probability) &&
        close(d.share.mean, mean) &&
        close(d.share.quantiles.p05, lower) &&
        close(d.share.quantiles.p95, upper),
      "Published prediction or observed winner differs.",
    );
    check(
      close(ae, Math.abs(mean - actual)) &&
        close(+s.squared_error, (mean - actual) ** 2) &&
        close(width, upper - lower) &&
        close(
          intervalScore,
          width +
            20 * (Math.max(lower - actual, 0) + Math.max(actual - upper, 0)),
        ),
      "Saved share error or interval score differs.",
    );
    const covered = lower <= actual && actual <= upper;
    check(
      ["True", "False"].includes(s.coverage) &&
        (s.coverage === "True") === covered,
      "Saved interval coverage differs.",
    );
    check(
      Number.isFinite(crps) &&
        crps >= 0 &&
        crps <= 1 &&
        close(
          +s.binary_brier,
          (d.winProbability - Number(actual > 0.5)) ** 2,
        ) &&
        close(+s.multiclass_brier, 2 * +s.binary_brier),
      "Saved probability score differs.",
    );
    totals.covered += Number(covered);
    totals.meanAbsoluteError += ae;
    totals.meanIntervalWidth += width;
    totals.crps += crps;
    totals.intervalScore += intervalScore;
  }
  for (const k of [
    "meanAbsoluteError",
    "meanIntervalWidth",
    "crps",
    "intervalScore",
  ])
    totals[k] /= totals.total;
  return totals;
}
export async function buildHouseScoreEvidence({
  sourceRoot,
  publicationDirectory,
}) {
  const sourceReal = await realpath(sourceRoot);
  async function pinned(relative, sha) {
    const p = await realpath(path.join(sourceRoot, relative));
    check(p.startsWith(sourceReal + path.sep), "Source leaves checkout.");
    const bytes = await readFile(p);
    check(digest(bytes) === sha, `Source pin changed: ${relative}`);
    return bytes;
  }
  const paths = {
    scoreReceipt: "final-scores-v2/receipt.json",
    perRaceScores: "final-scores-v2/per-race-scores.csv",
    cycleScores: "final-scores-v2/cycle-scores.csv",
    observedLabels: "panel-v6/outcomes-SEPARATE.csv",
    rosterFeatures: "panel-v6/features.csv",
    precisionReceipt: "precision-v1/receipt.json",
    reportReceipt: "report-v1/receipt.json",
  };
  const saved = {};
  for (const [key, relative] of Object.entries(paths))
    saved[key] = await pinned(
      `${BASE}/${relative}`,
      HOUSE_SCORE_EVIDENCE_SOURCE_PINS[key],
    );
  const receipt = JSON.parse(saved.scoreReceipt);
  check(
    receipt.schema === "house-score-receipt-v1" &&
      receipt.target_id === "house_aggregate_fundamentals_retrospective_v1" &&
      receipt.scope === "eligible_race_subset" &&
      receipt.full_chamber === false,
    "Score receipt scope differs.",
  );
  for (const pin of [
    ...receipt.sources,
    ...receipt.outputs,
    ...receipt.fit_receipts,
  ])
    await pinned(pin.path, pin.sha256);
  const comparison = await buildHouseComparisonManifest({
    sourceRoot,
    publicationDirectory,
  });
  const rows = parseCsv(saved.perRaceScores.toString()),
    cycles = parseCsv(saved.cycleScores.toString()),
    labels = parseCsv(saved.observedLabels.toString()),
    features = parseCsv(saved.rosterFeatures.toString());
  const publication = JSON.parse(
    await readFile(path.join(publicationDirectory, "publication.json"), "utf8"),
  );
  const catalog = JSON.parse(
    await readFile(
      path.join(publicationDirectory, publication.catalog.file),
      "utf8",
    ),
  );
  const entries = [];
  const used = new Set();
  for (const group of comparison.groups)
    for (const binding of group.runs) {
      check(
        receipt.fit_receipts.filter(
          (p) => p.sha256 === binding.fitReceiptSha256,
        ).length === 1,
        "Scoring fit receipt differs from reviewed result.",
      );
      const entry = catalog.runs.find((r) => r.id === binding.runId);
      const resultBytes = await readFile(
        path.join(publicationDirectory, entry.result.file),
      );
      check(
        digest(resultBytes) === binding.resultSha256,
        "Published result changed during evidence export.",
      );
      const result = JSON.parse(resultBytes);
      check(
        digest(houseComparisonCanonicalJson(result)) ===
          binding.canonicalResultSha256,
        "Canonical published result differs.",
      );
      const selected = rows.filter(
        (r) =>
          r.configuration === binding.configurationId &&
          +r.cycle === group.year,
      );
      selected.forEach((r) => used.add(r));
      const summary = validateHouseScoreRows(
        result,
        selected,
        labels,
        features,
      );
      const cycle = cycles.filter(
        (r) =>
          r.configuration === binding.configurationId &&
          +r.cycle === group.year &&
          r.subset === "all",
      );
      check(
        cycle.length === 1 &&
          cycle[0].numerical_status === "passed" &&
          +cycle[0].races === summary.total,
        "Missing unique all-race source summary.",
      );
      for (const [local, source] of Object.entries({
        meanAbsoluteError: "absolute_error",
        meanIntervalWidth: "interval_width",
        crps: "crps",
        intervalScore: "interval_score",
      }))
        check(
          close(summary[local], +cycle[0][source]),
          "Cycle score summary differs.",
        );
      check(
        close(summary.covered / summary.total, +cycle[0].coverage),
        "Cycle coverage differs.",
      );
      entries.push({
        runId: binding.runId,
        configurationId: binding.configurationId,
        configurationSha256: binding.configurationSha256,
        resultSha256: binding.resultSha256,
        canonicalResultSha256: binding.canonicalResultSha256,
        fitReceiptSha256: binding.fitReceiptSha256,
        year: group.year,
        mapVintage: group.mapVintage,
        raceIds: group.raceIds,
        raceCount: group.raceCount,
        units: "share_fraction",
        intervalLevel: 0.9,
        scope: "eligible_race_subset",
        ...summary,
      });
    }
  check(used.size === rows.length, "Unexpected scored model, year or race.");
  const manifest = {
    schemaVersion: 1,
    id: "reviewed-house-share-tests-20261020-v1",
    sourcePins: HOUSE_SCORE_EVIDENCE_SOURCE_PINS,
    qualifications: [
      "Reviewed experimental historical testing evidence, not a current forecast or strict as-of replay. Source editions and final candidate rosters are retrospective proxies.",
      "All metrics concern the included Democratic/Republican contests and D/(D+R) vote shares; excluded districts and third-party votes are outside these summaries.",
      "Coverage counts actual shares inside inclusive central 90% intervals. Coverage must be read with interval width and proper scores; wider intervals can cover more outcomes.",
      "CRPS and interval scores are preserved reviewed scores of the saved predictive distributions. Lower is better; no predictive arrays are reconstructed by the frontend.",
      "Districts share election shocks; four evaluation elections do not establish future calibration or generalization. No accuracy confidence interval or significance claim is supplied.",
      "Numerical race checks pass, while eligible-seat subtotal precision fails in all sixteen results. No chamber-control qualification follows.",
      "The 2018 Context model extrapolates beyond the presidential-party context present in its training elections.",
    ],
    entries,
  };
  check(
    parseHouseScoreEvidenceManifest(manifest),
    "Generated evidence manifest invalid.",
  );
  return manifest;
}
if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const args = process.argv.slice(2);
  const option = (name, fallback) => {
    const i = args.indexOf(name);
    return i < 0 ? fallback : args[i + 1];
  };
  const manifest = await buildHouseScoreEvidence({
    sourceRoot: option(
      "--source-root",
      "/home/hayden/projects/ballotmancy-house-benchmark",
    ),
    publicationDirectory: option(
      "--publication-dir",
      path.resolve("data/house"),
    ),
  });
  await writeFile(
    option("--output", "data/house-score-evidence.json"),
    JSON.stringify(manifest, null, 2) + "\n",
  );
  console.log(
    `Exported ${manifest.entries.length} reviewed share-test summaries. Entries SHA256 ${digest(houseComparisonCanonicalJson(manifest.entries))}`,
  );
}
