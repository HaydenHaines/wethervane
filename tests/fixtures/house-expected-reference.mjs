import { readFileSync } from "node:fs";

const raw = JSON.parse(readFileSync(new URL("./election-night/house-compatibility-sidecar.json", import.meta.url)));
const h = digit => digit.repeat(64);
export function makeCompatibilityHouseReference(forecast) {
  // This wraps a compatibility-only sidecar with invented review pins for tests.
  const reference = structuredClone(raw);
  const sourcePins = {
    collectionSha256: h("a"), collectionReceiptSha256: h("b"),
    nativeCatalogSha256: reference.catalog.sha256, readyPacketSha256: reference.ready_packet.sha256,
    referenceSha256: h("c"), sidecarReceiptSha256: h("d"), reviewSha256: h("e"), auditorSha256: h("f"),
    sourceReceiptSha256: reference.source_receipt.sha256, jointDrawsSha256: reference.joint_draws.sha256,
    sourcePlanSha256: reference.source_plan.sha256, conditionalSharesSha256: reference.conditional_shares.sha256,
    jointMetricArchiveSha256: reference.joint_metric_archive.sha256,
    scientificReviewSha256: reference.scientific_review.sha256,
    nativeNumericalReviewSha256: reference.native_numerical_review.sha256,
    qualificationSha256: reference.qualification.sha256, selectionSha256: reference.selection.sha256,
    builderSha256: reference.builder.sha256, operatorReviewSha256: h("1"),
  };
  const row = { chamber: "house", runId: forecast.run.id, resultSha256: forecast.run.artifactSha256,
    configuration: { ...forecast.run.configuration }, forecastIssuedAt: forecast.run.issuedAt,
    forecastCutoffAt: forecast.run.cutoffAt, referenceIssuedAt: reference.issued_at,
    rows: reference.draw_layout.rows, chains: reference.draw_layout.chains,
    winnerRaceCount: 435, pairRaceCount: 412, pairInapplicableRaceCount: 23,
    sourcePins, reference };
  const catalog = { schema: "HouseExpectedErrorPublicCatalogV1", schemaVersion: 1, chamber: "house",
    defaultRunId: forecast.run.id, sourceCatalogSha256: sourcePins.collectionSha256,
    nativeCatalogSha256: sourcePins.nativeCatalogSha256, readyPacketSha256: sourcePins.readyPacketSha256,
    evaluationLock: false, observedResultsConsumed: false, entries: [row] };
  const { reference: unused, ...qualification } = row;
  return { catalog, qualification };
}
