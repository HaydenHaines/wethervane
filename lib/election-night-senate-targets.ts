/** Stable observation-selection identifiers, not official feed or tabulation IDs.
 * The admitted observation retains the actual source, stage, date and round.
 * The policy operator must qualify that observation before using these identifiers.
 */
import { SENATE_PAIR_RESULT_POLICY, type SenateResultPolicyRow } from './election-night-senate-results.ts';

export interface SenateResultSelector {
  eventId: string;
  roundId: string;
  resultRuleId: string;
  ruleFamily: null;
}

export function senateResultSelector(row: SenateResultPolicyRow): SenateResultSelector {
  if (!row || typeof row.raceId !== 'string' || !row.raceId || row.generalDate !== '2026-11-03' ||
    !['plurality_named_pair', 'rcv_locked_final_two', 'ga_certified_decisive_event'].includes(row.operatorFamily))
    throw Error('Senate result selector requires an exact reviewed 2026 policy row.');
  const runoff = row.operatorFamily === 'ga_certified_decisive_event';
  if ((row.state === 'GA') !== runoff ||
    (['AK', 'ME'].includes(row.state)) !== (row.operatorFamily === 'rcv_locked_final_two'))
    throw Error('The state must retain its reviewed plurality, ranked-choice or runoff policy.');
  if (runoff && (row.state !== 'GA' || row.runoffDate !== '2026-12-01'))
    throw Error('Georgia selector requires the reviewed state and runoff date.');
  if (row.operatorFamily === 'rcv_locked_final_two' && !['AK', 'ME'].includes(row.state))
    throw Error('Ranked-choice selector requires the reviewed Alaska or Maine row.');
  return {
    eventId: `${row.raceId}:${runoff ? 'decisive_general_or_qualifying_runoff_v1' : 'general_2026-11-03'}`,
    roundId: runoff ? 'certified_no_runoff_general_or_exact_pair_runoff_v1'
      : row.operatorFamily === 'rcv_locked_final_two' ? 'official_observed_exact_locked_final_two_v1'
        : 'certified_decisive_general_v1',
    resultRuleId: `senate-pair:${SENATE_PAIR_RESULT_POLICY.policySha256}:${row.operatorFamily}`,
    // Original public Senate forecasts do not declare a ruleFamily. Preserve that field.
    ruleFamily: null,
  };
}
