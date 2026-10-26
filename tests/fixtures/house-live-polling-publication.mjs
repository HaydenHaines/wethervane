import { SYNTHETIC_SHA } from "./house-live-fixtures.mjs";
import { HOUSE_LIVE_POLL_RULESET, parseHouseLivePollingRun } from "../../lib/house-live-polling.ts";

/** Invented browser data only. This does not authenticate native artifacts or admit publication. */
export function withInventedPolling(input) {
  const publication = structuredClone(input), { view } = publication;
  const enabled = view.run.configuration.polling === "standard";
  const questions = ["AL-01", "AL-01", "AL-04"].map((district, index) => ({
    observationId: `invented-question-${index}`, raceId: `2026:${district}:house:regular`,
    pollster: `Invented Research ${index + 1}`, startDate: "2026-10-10", fieldEnd: "2026-10-12",
    availableAt: "2026-10-14T00:00:00Z", population: index === 0 ? "LV" : "RV",
    sampleSize: index === 1 ? null : 650, sourceId: "invented-poll-source", sourceSha256: SYNTHETIC_SHA,
    sourceUrls: [`https://example.invalid/invented-poll-${index}`],
  }));
  const excluded = [
    ...Array.from({ length: 12 }, (_, index) => ({ districtKey: "AL-01", pollster: `Invented Old Poll ${index + 1}`,
      endDate: "2026-09-21", reason: "earlier_same_pollster_race_sample" })),
    { districtKey: "AL-03", pollster: "Invented Primary Poll", endDate: "2026-04-03", reason: "not_2026_house_regular_general" },
    { districtKey: null, pollster: "Record without a current district", endDate: null, reason: "not_one_current_house_district" },
  ];
  const excludedReasonCounts = [
    { districtKey: null, reason: "not_one_current_house_district", count: 1 },
    { districtKey: "AL-01", reason: "earlier_same_pollster_race_sample", count: 12 },
    { districtKey: "AL-03", reason: "not_2026_house_regular_general", count: 1 },
  ];
  view.sources.push({ id: "invented-poll-source", label: "Invented polling source, not election data",
    url: "https://example.invalid/invented-polls", retrieved_at: "2026-10-19T00:00:00Z",
    sha256: SYNTHETIC_SHA, role: "polling" });
  if (!view.qualification.evidenceHashes.includes(SYNTHETIC_SHA)) view.qualification.evidenceHashes.push(SYNTHETIC_SHA);
  for (const race of view.races) race.pollCount = enabled ? questions.filter(question => question.raceId === race.id).length : 0;
  publication.polling = {
    schema: "HouseLivePollingRunV1", runId: view.run.id, resultSha256: view.run.resultSha256,
    ruleset: HOUSE_LIVE_POLL_RULESET,
    cutoffs: { forecastAt: view.run.cutoffAt, pollingAt: "2026-10-20T00:00:00Z" },
    bindings: { catalogSha256: SYNTHETIC_SHA, sourcePolicySha256: SYNTHETIC_SHA,
      admissionSha256: SYNTHETIC_SHA, selectionSha256: SYNTHETIC_SHA, ledgerSha256: SYNTHETIC_SHA },
    modelUsesSelectedQuestions: enabled, usedQuestions: enabled ? questions : [],
    selectedPoolQuestionCount: 3, selectedPoolDistrictCount: 2, excluded, excludedReasonCounts,
  };
  if (!parseHouseLivePollingRun(publication.polling)) throw Error("Invalid invented browser polling fixture");
  return publication;
}
