/** Research-only arithmetic verifier for an explicitly weighted joint stream.
 * The caller must independently authenticate source bytes, weight/row pairing,
 * input admission, and scientific/numerical qualification. No publication.
 */
type Party = 'D' | 'R' | 'O';
type Quantile = 'p05' | 'p10' | 'p50' | 'p90' | 'p95';
interface Race { race_id: string; outcomes: { outcome_id: Party; p_win: number }[] }
interface Distribution { expected: number; pmf: number[]; quantiles: Record<Quantile, number> }
export interface WeightedResearchResult {
  races: Race[];
  joint_draws: { count: number; chain_count: number; ordered_race_ids: string[];
    winner_encoding: Record<Party, number> };
  chamber: null | { seats: Record<Party, Distribution>; majority: { D: number; R: number; unresolved: number } };
}
export interface WeightedRow {
  jointDrawId: number;
  predictiveChain: number;
  predictiveDraw: number;
  winnerCodes: readonly number[];
  logFactor: number;
}
export interface WeightedCheck {
  rowsChecked: number;
  logWeightSum: number;
  kishEss: number;
  maxNormalizedWeight: number;
  chamberChecked: boolean;
  quantileAmbiguities: { group: Party; quantile: number; lower: number; upper: number }[];
}

const PARTIES = ['D', 'R', 'O'] as const;
const QUANTILES: [Quantile, number][] = [['p05', .05], ['p10', .1], ['p50', .5], ['p90', .9], ['p95', .95]];
const PROBABILITY_TOLERANCE = 1e-8;
const SEAT_TOLERANCE = 1e-6;
const QUANTILE_BOUNDARY_TOLERANCE = 1e-12;

function requireCheck(value: unknown, message: string): asserts value {
  if (!value) throw new Error(message);
}
function near(actual: number, expected: number, tolerance: number): boolean {
  return Number.isFinite(actual) && Number.isFinite(expected) && Math.abs(actual - expected) <= tolerance;
}
function inverseCdf(histogram: Float64Array, total: number, q: number): number {
  let mass = 0;
  for (let i = 0; i < histogram.length; i++) {
    mass += histogram[i];
    if (mass >= q * total) return i;
  }
  throw new Error('Weighted seat histogram cannot supply the requested quantile.');
}

/** Weighted inverse CDF with explicit floating-boundary flags; no release gate. */
export async function checkWeightedJointRows(result: WeightedResearchResult,
  rows: AsyncIterable<WeightedRow> | Iterable<WeightedRow>): Promise<WeightedCheck> {
  const joint = result.joint_draws;
  requireCheck(result.races.length === 435 && joint.ordered_race_ids.length === 435 &&
    Number.isSafeInteger(joint.count) && joint.count > 0 && Number.isSafeInteger(joint.chain_count) &&
    joint.chain_count > 0 && joint.count % joint.chain_count === 0 &&
    PARTIES.every((p, i) => joint.winner_encoding[p] === i), 'Invalid ordered joint layout.');
  const raceIds = new Set<string>();
  const support = result.races.map((race, i) => {
    requireCheck(typeof race.race_id === 'string' && race.race_id.length > 0 && !raceIds.has(race.race_id)
      && joint.ordered_race_ids[i] === race.race_id, 'Race identities or order differ.');
    raceIds.add(race.race_id);
    const groups = new Set(race.outcomes.map(x => x.outcome_id));
    requireCheck(groups.size === race.outcomes.length && groups.size > 0 &&
      [...groups].every(x => PARTIES.includes(x)), 'Invalid race support.');
    return groups;
  });
  const raceMass = Array.from({ length: 435 }, () => new Float64Array(3));
  const seatMass = PARTIES.map(() => new Float64Array(436));
  const controlMass = new Float64Array(3);
  let rowsChecked = 0, maxLog = -Infinity, total = 0, squared = 0;
  const perChain = joint.count / joint.chain_count;
  for await (const row of rows) {
    requireCheck(rowsChecked < joint.count && row.jointDrawId === rowsChecked &&
      row.predictiveChain === Math.floor(rowsChecked / perChain) && row.predictiveDraw === rowsChecked % perChain,
      'Weighted row identity or chain ordering differs.');
    requireCheck(Number.isFinite(row.logFactor) && Array.isArray(row.winnerCodes) && row.winnerCodes.length === 435,
      'Finite log factor and 435 winner codes required.');
    if (row.logFactor > maxLog) {
      const scale = maxLog === -Infinity ? 0 : Math.exp(maxLog - row.logFactor);
      total *= scale; squared *= scale * scale;
      for (const counts of raceMass) for (let k = 0; k < 3; k++) counts[k] *= scale;
      for (const histogram of seatMass) for (let k = 0; k <= 435; k++) histogram[k] *= scale;
      for (let k = 0; k < 3; k++) controlMass[k] *= scale;
      maxLog = row.logFactor;
    }
    const w = Math.exp(row.logFactor - maxLog);
    total += w; squared += w * w;
    const seats = [0, 0, 0];
    for (let i = 0; i < 435; i++) {
      const code = row.winnerCodes[i];
      requireCheck(Number.isInteger(code) && code >= 0 && code <= 2 && support[i].has(PARTIES[code]),
        `Invalid or unsupported winner at race ${i}.`);
      raceMass[i][code] += w;
      seats[code]++;
    }
    requireCheck(seats[0] + seats[1] + seats[2] === 435, 'Seat conservation failed.');
    for (let k = 0; k < 3; k++) seatMass[k][seats[k]] += w;
    controlMass[seats[0] >= 218 ? 0 : seats[1] >= 218 ? 1 : 2] += w;
    rowsChecked++;
  }
  requireCheck(rowsChecked === joint.count && Number.isFinite(total) && total > 0 && squared > 0,
    'Weighted row count or mass differs.');
  for (let i = 0; i < 435; i++) for (const outcome of result.races[i].outcomes) {
    requireCheck(near(outcome.p_win, raceMass[i][joint.winner_encoding[outcome.outcome_id]] / total, PROBABILITY_TOLERANCE),
      `Weighted winner mass differs for race ${result.races[i].race_id}.`);
  }
  const quantileAmbiguities: WeightedCheck['quantileAmbiguities'] = [];
  if (result.chamber !== null) {
    for (let k = 0; k < 3; k++) {
      const published = result.chamber.seats[PARTIES[k]], histogram = seatMass[k];
      requireCheck(Array.isArray(published.pmf) && published.pmf.length === 436, 'Invalid seat PMF.');
      let mean = 0;
      for (let seats = 0; seats <= 435; seats++) {
        const p = histogram[seats] / total;
        requireCheck(near(published.pmf[seats], p, PROBABILITY_TOLERANCE), 'Weighted seat PMF differs.');
        mean += seats * p;
      }
      requireCheck(near(published.expected, mean, SEAT_TOLERANCE), 'Weighted seat mean differs.');
      for (const [name, q] of QUANTILES) {
        const lower = inverseCdf(histogram, total, q - QUANTILE_BOUNDARY_TOLERANCE);
        const upper = inverseCdf(histogram, total, q + QUANTILE_BOUNDARY_TOLERANCE);
        requireCheck(near(published.quantiles[name], lower, SEAT_TOLERANCE), 'Weighted inverse-CDF seat quantile differs.');
        if (lower !== upper) quantileAmbiguities.push({ group: PARTIES[k], quantile: q, lower, upper });
      }
    }
    const m = result.chamber.majority;
    requireCheck(near(m.D, controlMass[0] / total, PROBABILITY_TOLERANCE) &&
      near(m.R, controlMass[1] / total, PROBABILITY_TOLERANCE) &&
      near(m.unresolved, controlMass[2] / total, PROBABILITY_TOLERANCE), 'Weighted majority mass differs.');
  }
  return { rowsChecked, logWeightSum: maxLog + Math.log(total), kishEss: total * total / squared,
    maxNormalizedWeight: 1 / total, chamberChecked: result.chamber !== null, quantileAmbiguities };
}
