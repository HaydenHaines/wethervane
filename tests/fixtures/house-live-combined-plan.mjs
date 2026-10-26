import { createHash } from "node:crypto";

// Invented lineage only. The source files do not exist; tests establish structure,
// plan authentication and row identity, never scientific source-run equivalence.
const pin = path => ({ path, sha256: createHash("sha256").update(`invented:${path}`).digest("hex") });

export function makeHouseLiveCombinedPlan(result, sourceRunLengths = [2, 2]) {
  const draws = result.joint_draws.count / result.joint_draws.chain_count;
  let jointOffset = 0, chainOffset = 0;
  const source_runs = sourceRunLengths.map((length, index) => {
    if (!Number.isInteger(length / draws)) throw Error("Invented sources must contain whole chains.");
    const directory = `invented-source-${index}`;
    const row = {
      directory,
      receipt: pin(`research/house-live-20261025/${directory}/receipt.json`),
      plan: pin(`research/house-live-20261025/${directory}/plan.json`),
      settings: { chains: length / draws, draws, warmup: 1, seed: 2000 + index },
      joint_draw_offset: jointOffset,
      predictive_chain_offset: chainOffset,
    };
    jointOffset += length;
    chainOffset += row.settings.chains;
    return row;
  });
  return {
    schema: "HouseLiveCombinedPredictionPlanV1", issued_at: result.issued_at,
    configuration: { id: result.id, ...result.configuration }, pilot: false,
    settings: { chains: result.joint_draws.chain_count, draws, warmup_per_source: 1 },
    source_runs,
    inputs: [pin("research/house-live-20261025/invented-input.json"), ...source_runs.map(row => row.receipt)],
    source: pin("research/house-live-20261025/invented-combined/run-source.py"),
    polling_error_scales: { independent: 0.01, cycle: 0.01, race: 0.02, house: 0.01, movement: 0.01 },
    exact_electoral_rule_simulation: false, source_2026_outcomes_loaded: false,
    combination: "Invented complete source rows, retaining each source chain, for tests only.",
  };
}
