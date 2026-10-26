#!/usr/bin/env node
/**
 * Deterministic, non-admitted pre-outcome Senate lock proposal. No actual results,
 * trusted publication/admission manifests, or scientific review is created here.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { ElectionNightArtifactStore } from "../lib/election-night-artifacts.ts";
import { toSenateElectionNightForecast } from "../lib/election-night-forecast.ts";
import { matchElectionNightCoreReference } from "../lib/election-night-core-reference.ts";
import { ELECTION_NIGHT_RUBRIC, gradeElectionNightRubric } from "../lib/election-night-rubric.ts";
import { senateResultSelector } from "../lib/election-night-senate-targets.ts";

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const OWNER = "/home/hayden/projects/wethervane-redesign-20261006/integration";
const DEFAULT_OUTPUT = "/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/election-night-senate-lock-draft-sol-v1";
const ownerBase = "research/senate-polling-refresh-20261026";
const e = ELECTION_NIGHT_RUBRIC.earlierBenchmark;
const rawPins = {
  registry: "c1e6d85bd4886768975d91d45f57a151114564dfb8f73645b484f089a5214031",
  resultMapping: e.resultMappingSha256,
  policy: e.resultPolicySha256,
  comparisonReview: e.reviewSha256,
  mapping: e.mappingSha256,
  mappingFreeze: e.freezeManifestSha256,
  resultPolicyFreeze: e.resultPolicyFreezeSha256,
  fixed32Handoff: "e571037c39a1203b79c2027ec72a4e2e7ff3b3340c1648eba39f75b6a7c41fbb",
  fixed32Review: "b82bc61967c3af40274178d79f65ba088e3f5313d32233ec4fbc0c9ac3648c8d",
  fixed32ReviewFreeze: "a90c5c30a2db502a71ff56b98c5c2fc7966f9b1675539fc30b479d9bca18f3d3",
  coreReferenceCatalog: "8a63d2d3f8490b991a3da6b92ad686d2f9a5d0eb3dd03d51f0c644d8d3e16407",
  coreReferencePublication: "af4c54bc163625f842014a097837640670512c4af3737b69be182392c1a03260",
  full35ReferenceCatalog: "9d2c289667d9f25fe38515eefb1503133a6f72b92cfcf65e5061540b235ccd5f",
};
const ownerFiles = {
  resultMapping: `${ownerBase}/result-policy-special-rules-sol-v1/locked-35-result-policy-mapping-v1.json`,
  policy: `${ownerBase}/result-policy-special-rules-sol-v1/policy-v1.json`,
  comparisonReview: `${ownerBase}/scorecard-baseline-compatibility-sol/review-v1.json`,
  mapping: `${ownerBase}/scorecard-baseline-compatibility-sol/candidate-pair-event-mapping-v1.json`,
  mappingFreeze: `${ownerBase}/scorecard-baseline-compatibility-sol/freeze-manifest-v1.json`,
  resultPolicyFreeze: `${ownerBase}/result-policy-special-rules-sol-v1/freeze-manifest-v1.json`,
  fixed32Handoff: `${ownerBase}/fixed32-scorecard-frontend-handoff-root-v1.json`,
  fixed32Review: `${ownerBase}/fixed32-scorecard-review-sol-v1/review-v1.json`,
  fixed32ReviewFreeze: `${ownerBase}/fixed32-scorecard-review-sol-v1/freeze-v1.json`,
};
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
function need(value, reason) { if (!value) throw Error(`Senate lock draft: ${reason}`); }
function same(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
function within(root, path) {
  need(!isAbsolute(path) && !path.split("/").includes(".."), `Unsafe source path: ${path}`);
  const resolvedRoot = realpathSync(root), resolved = realpathSync(resolve(resolvedRoot, path)),
    rel = relative(resolvedRoot, resolved);
  need(rel && rel !== ".." && !rel.startsWith("../") && !isAbsolute(rel), `Unsafe source path: ${path}`);
  return resolved;
}
export function readPinnedFile(root, path, expectedSha256) {
  need(/^[a-f0-9]{64}$/.test(expectedSha256), "A reviewed byte hash is required.");
  const bytes = readFileSync(within(root, path));
  need(sha(bytes) === expectedSha256, `Source bytes changed: ${path}`);
  return bytes;
}
function json(bytes) { return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)); }
function artifact(store, bytes) { return store.put(bytes); }
function ownerArtifact(store, root, key) { return artifact(store, readPinnedFile(root, ownerFiles[key], rawPins[key])); }
function policyRow(row) {
  return { raceId: row.race_id, state: row.state, seatId: row.seat_id, eventKind: row.event_kind,
    generalDate: row.scheduled_general_event_date, runoffDate: row.scheduled_runoff_event_date,
    firstCandidateId: row.first_candidate_id, secondCandidateId: row.second_candidate_id,
    operatorFamily: row.operator_family };
}
function protocol(forecast, selectors) {
  return { schema: "ElectionNightEvaluationProtocolV1", id: `senate2026-${forecast.run.id}-pair-policy-draft-v1`,
    forecast: { chamber: "senate", runId: forecast.run.id, artifactSha256: forecast.run.artifactSha256,
      issuedAt: forecast.run.issuedAt, cutoffAt: forecast.run.cutoffAt },
    // A separate reviewer must bind the exact projected bytes and selector mapping.
    reviewedDefinition: null,
    races: forecast.races.map((race, i) => {
      const selector = selectors[i].selector, pair = race.pair;
      need(pair && race.ruleFamily === null, `Missing frozen Senate pair for ${race.id}`);
      return { raceId: race.id,
        winner: { estimand: "conditional_principal_pair_winner", eventId: selector.eventId, roundId: selector.roundId,
          ruleFamily: null, resultRuleId: selector.resultRuleId, outcomeIds: race.outcomes.map(outcome => outcome.id), minimumStatus: "certified" },
        margin: { estimand: "named_pair_margin", eventId: selector.eventId, roundId: selector.roundId,
          ruleFamily: null, resultRuleId: selector.resultRuleId, firstCandidateId: pair.firstCandidateId,
          secondCandidateId: pair.secondCandidateId, denominator: pair.denominator, units: "percentage_points",
          orientation: "first_minus_second", pointRule: "mean", minimumStatus: "certified" } };
    }) };
}
function targets(selectors, coreIds, forecast, denominator) {
  const byId = new Map(forecast.races.map(race => [race.id, race]));
  const bySelector = new Map(selectors.map(row => [row.policyRow.raceId, row.selector]));
  const winners = [], intervals = [];
  for (const raceId of coreIds) {
    const race = byId.get(raceId), selector = bySelector.get(raceId), pair = race?.pair;
    need(race && selector && pair, `Core pair missing: ${raceId}`);
    const base = { raceId, eventId: selector.eventId, roundId: selector.roundId,
      resultRuleId: selector.resultRuleId, denominator, firstCandidateId: pair.firstCandidateId,
      secondCandidateId: pair.secondCandidateId };
    winners.push({ ...base, estimand: "senate_principal_pair_lead", outcomeIds: [pair.firstCandidateId, pair.secondCandidateId] });
    intervals.push({ ...base, estimand: "named_pair_margin", units: "percentage_points",
      orientation: "first_minus_second", intervalLevel: 0.9 });
  }
  return { winners, intervals };
}
function writeExact(path, bytes) {
  if (existsSync(path)) need(readFileSync(path).equals(bytes), `Refusing to replace prior proposal: ${path}`);
  else writeFileSync(path, bytes);
}
/** Return a reviewable artifact graph; nothing here is a trusted admission or an actual result. */
export function prepareSenateLockDraft({ repoRoot = REPO, ownerRoot = OWNER, outputDir = DEFAULT_OUTPUT } = {}) {
  const repo = resolve(repoRoot), owner = resolve(ownerRoot), output = resolve(outputDir);
  need(owner === resolve(OWNER), "Unreviewed scientific owner root.");
  need(!output.startsWith(`${repo}/public/`) && !output.startsWith(`${repo}/data/`) &&
    !output.startsWith(`${owner}/`), "Draft output must stay outside public and scientific source data.");
  mkdirSync(output, { recursive: true });
  const store = new ElectionNightArtifactStore(output);
  const registryBytes = readPinnedFile(repo, "public/data/explorer/registry.json", rawPins.registry), registry = json(registryBytes);
  need(registry.schemaVersion === 1 && Array.isArray(registry.runs), "Unexpected forecast registry.");
  const modelBinding = { chamber: "senate", ...e.model }, baselineBinding = { chamber: "senate", ...e.baseline };
  const original = {}, projection = {}, sourcePin = {}, projectionPin = {}, chosenEntry = {};
  for (const [role, binding] of [["model", modelBinding], ["baseline", baselineBinding]]) {
    const entry = registry.runs.find(row => row.id === binding.runId);
    need(entry && entry.sha256 === binding.resultSha256 && entry.targetId === binding.targetId &&
      entry.lawId === binding.lawId && entry.issuedAt === binding.issuedAt && entry.cutoffAt === binding.cutoffAt,
    `Registry differs from reviewed ${role} forecast identity.`);
    const bytes = readPinnedFile(repo, `public/data/explorer/${binding.runId}.json`, binding.resultSha256);
    const forecast = toSenateElectionNightForecast(json(bytes), entry);
    need(forecast.run.artifactSha256 === binding.resultSha256 &&
      forecast.sourceHashes.forecastSha256 === binding.nativeForecastSha256 && forecast.races.length === 35,
    `Projected ${role} forecast differs from the reviewed edition.`);
    original[role] = bytes; projection[role] = forecast; chosenEntry[role] = entry;
    sourcePin[role] = artifact(store, bytes); projectionPin[role] = store.putJson(forecast);
  }
  const ownerPins = Object.fromEntries(Object.keys(ownerFiles).map(key => [key, ownerArtifact(store, owner, key)]));
  const policy = json(store.read(ownerPins.policy)), mapping = json(store.read(ownerPins.resultMapping));
  need(policy.schema === "Senate35PreResultOperatorPolicyV1" && mapping.schema === "LockedSenate35PreResultMappingV1" &&
    mapping.race_count === 35, "Owner result-policy mapping changed.");
  const fixed = policy.frontend_binding.fixed_cohorts, fullIds = fixed.full_35_locked_rows, coreIds = fixed.plurality_core_32,
    conditionalIds = fixed.special_conditional_3;
  need(Array.isArray(fullIds) && fullIds.length === 35 && Array.isArray(coreIds) && coreIds.length === 32 &&
    same(conditionalIds, e.conditionalRaceIds) && same(fullIds, projection.model.races.map(r => r.id)) &&
    same(fullIds, projection.baseline.races.map(r => r.id)) && same(mapping.races.map(r => r.race_id), fullIds) &&
    same(coreIds, fullIds.filter(id => !conditionalIds.includes(id))), "Reviewed full35/core32 roster differs.");
  const selectors = mapping.races.map((raw, i) => {
    const row = policyRow(raw), model = projection.model.races[i], baseline = projection.baseline.races[i];
    need(raw.official_future_feed_candidate_ids === null && raw.official_future_feed_contest_id === null &&
      raw.future_result_crosswalk_status.startsWith("requires_separate_review") &&
      row.raceId === model.id && row.raceId === baseline.id && row.seatId === model.seatId && row.seatId === baseline.seatId &&
      row.firstCandidateId === model.pair.firstCandidateId && row.secondCandidateId === model.pair.secondCandidateId &&
      row.firstCandidateId === baseline.pair.firstCandidateId && row.secondCandidateId === baseline.pair.secondCandidateId &&
      row.generalDate === "2026-11-03" && ["plurality_named_pair", "rcv_locked_final_two", "ga_certified_decisive_event"].includes(row.operatorFamily),
    `Reviewed result selector identity changed for ${row.raceId}.`);
    return { policyRow: row, selector: senateResultSelector(row), semanticRole: "predeclared_observation_selection_rule" };
  });
  const cohort = { schema: "ElectionNightSenateCohortProposalV1", status: "requires_independent_admission_review",
    policySha256: rawPins.policy, mappingSha256: rawPins.resultMapping,
    compatibilityMappingSha256: rawPins.mapping, fullDisplayRaceIds: fullIds,
    primaryGradeRaceIds: coreIds, ungradedDisplayRaceIds: conditionalIds,
    senateObservationSelectors: selectors,
    resultSourceBoundary: "No future official feed contest/candidate IDs, actual result source versions or observations are admitted." };
  const cohortPin = store.putJson(cohort), modelProtocol = protocol(projection.model, selectors),
    baselineProtocol = protocol(projection.baseline, selectors), modelProtocolPin = store.putJson(modelProtocol),
    baselineProtocolPin = store.putJson(baselineProtocol);
  const targetRows = targets(selectors, coreIds, projection.model,
    "sum_of_named_principal_candidate_votes_under_declared_rule_proxy");
  const identityPin = store.putJson({ schema: "ElectionNightComparisonIdentityProposalV1", id: e.primaryComparisonContractId,
    model: modelBinding, baseline: baselineBinding, cohortSha256: cohortPin.sha256,
    policySha256: rawPins.policy, winnerCohort: targetRows.winners, intervalCohort: targetRows.intervals });
  const rubricBytes = readPinnedFile(repo, "data/election-night-rubric-decision.json", ELECTION_NIGHT_RUBRIC.decisionSha256),
    rubricPin = artifact(store, rubricBytes);
  const comparison = { schema: "ElectionNightRubricComparisonV1", id: e.primaryComparisonContractId,
    sha256: identityPin.sha256, rubricDecisionSha256: rubricPin.sha256, comparisonMode: "explicit_earlier_benchmark",
    label: "October 26 polling vs October 20 electoral history, fixed 32 plurality pairs",
    model: modelBinding, baseline: baselineBinding,
    comparisonReview: { id: "reviewed-native-pair-compatibility", sha256: rawPins.comparisonReview },
    cohortDefinition: { id: "senate2026-full35-display-primary32-selectors-proposal-v1", sha256: cohortPin.sha256 },
    resultPolicy: { id: "senate35-pre-result-operator-policy-v1", sha256: rawPins.policy },
    earlierBenchmarkPins: { mappingSha256: rawPins.mapping, freezeManifestSha256: rawPins.mappingFreeze,
      resultMappingSha256: rawPins.resultMapping, resultPolicyFreezeSha256: rawPins.resultPolicyFreeze },
    displayRaceIds: fullIds, ungradedDisplayRaceIds: conditionalIds, houseMappingReviewSha256: null,
    winnerCohort: targetRows.winners, intervalCohort: targetRows.intervals, intervalExclusions: [] };
  const comparisonPin = store.putJson(comparison);
  const pending = gradeElectionNightRubric({ contract: comparison, modelForecast: modelBinding,
    baselineForecast: baselineBinding, winnerRows: [], intervalRows: [] });
  need(pending.status === "not_graded" && pending.issues.length === 0 && pending.winner.coverage.total === 32 &&
    pending.interval.coverage.total === 32, "Proposed rubric contract or locked cohort failed validation.");
  const coreBytes = readPinnedFile(repo, "public/data/election-night/core-references.json", rawPins.coreReferenceCatalog),
    coreCatalog = json(coreBytes), corePin = artifact(store, coreBytes), coreMatches = {};
  for (const role of ["model", "baseline"]) {
    const matched = matchElectionNightCoreReference(projection[role], coreCatalog);
    need(matched && matched.perRace.length === 32 && same(matched.perRace.map(row => row.raceId), coreIds) &&
      matched.evaluationLock === false && matched.observedResultsConsumed === false,
    `Fixed32 reference does not match the exact ${role} forecast.`);
    coreMatches[role] = { referenceSha256: matched.sourcePins.referenceSha256,
      jointDrawsSha256: matched.sourcePins.jointDrawsSha256, cohortSha256: matched.cohortSha256 };
  }
  const corePublicationPin = artifact(store, readPinnedFile(repo, "data/election-night-core-reference-publication.json", rawPins.coreReferencePublication));
  const full35ReferencePin = artifact(store, readPinnedFile(repo, "public/data/election-night/references.json", rawPins.full35ReferenceCatalog));
  const draft = { schema: "ElectionNightSenatePreOutcomeLockProposalV1", status: "awaiting_independent_review_and_admission",
    admissionIssued: false, actualResultsUsed: false, actualGradeAssigned: false,
    forecastBindings: { model: modelBinding, baseline: baselineBinding },
    lockArtifactCandidates: { modelSource: sourcePin.model, baselineSource: sourcePin.baseline,
      modelForecast: projectionPin.model, baselineForecast: projectionPin.baseline,
      modelProtocol: modelProtocolPin, baselineProtocol: baselineProtocolPin,
      comparisonContract: comparisonPin, cohort: cohortPin, rubric: rubricPin, resultPolicy: ownerPins.policy },
    comparisonIdentityProposal: identityPin,
    ownerEvidence: { registry: artifact(store, registryBytes), ...ownerPins,
      coreReferenceCatalog: corePin, coreReferencePublication: corePublicationPin,
      preservedFull35ReferenceCatalog: full35ReferencePin },
    fixed32ReferenceBindings: coreMatches,
    requiredIndependentReviews: ["exact original-to-normalized projection bytes for F and FP",
      "new all35 selector IDs, certified protocols and primary32 comparison/cohort artifact",
      "candidate lock manifest and scientific admission decision"],
    candidateLock: { schema: "ElectionNightEvaluationLockV1", id: "senate2026-F20261020-FP20261026-primary32-draft-v1",
      frozenAt: null, forecastBindings: { model: modelBinding, baseline: baselineBinding },
      artifacts: { modelSource: sourcePin.model, baselineSource: sourcePin.baseline,
        modelForecast: projectionPin.model, baselineForecast: projectionPin.baseline,
        modelProtocol: modelProtocolPin, baselineProtocol: baselineProtocolPin,
        comparisonContract: comparisonPin, cohort: cohortPin, rubric: rubricPin, resultPolicy: ownerPins.policy },
      reviews: null, projections: { model: { sourceSha256: sourcePin.model.sha256,
        projectionSha256: projectionPin.model.sha256, reviewSha256: null },
        baseline: { sourceSha256: sourcePin.baseline.sha256,
          projectionSha256: projectionPin.baseline.sha256, reviewSha256: null } } },
    emptyLedgerProposal: { schema: "ElectionNightResultLedgerV1", id: "senate2026-primary32-results-v1",
      evaluationLockSha256: null, chosenHeadSha256: null, revisions: [] },
    explicitlyMissing: ["protocol reviewedDefinition pins", "lock review-role pins and frozenAt",
      "trusted publication/admission manifests", "official result records and result-source admission"] };
  const draftBytes = Buffer.from(`${JSON.stringify(draft, null, 2)}\n`);
  writeExact(join(output, "draft.json"), draftBytes);
  const emptyLedgerEvidence = { schema: "ElectionNightEmptyLedgerEvidenceProposalV1",
    status: "requires_independent_lock_and_ledger_admission", draftSha256: sha(draftBytes),
    ledger: draft.emptyLedgerProposal,
    evidencePins: { resultPolicy: ownerPins.policy, resultMapping: ownerPins.resultMapping,
      cohort: cohortPin, comparisonContract: comparisonPin, originalModel: sourcePin.model,
      originalBaseline: sourcePin.baseline, nativePairReview: ownerPins.comparisonReview },
    actualSourcePins: [], actualResultRecords: [], trustedAdmissionIssued: false };
  writeExact(join(output, "empty-ledger-evidence.json"), Buffer.from(`${JSON.stringify(emptyLedgerEvidence, null, 2)}\n`));
  return draft;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const index = process.argv.indexOf("--output");
  need(index < 0 || process.argv[index + 1], "--output needs a path.");
  const draft = prepareSenateLockDraft({ outputDir: index < 0 ? DEFAULT_OUTPUT : process.argv[index + 1] });
  process.stdout.write(`Prepared ${draft.status} with 35 display rows, 32 primary targets and no actual results.\n`);
}
