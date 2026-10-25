/** Separately reviewed polling adapter. The predecessor native validator is unchanged. */
export const HOUSE_POLLING_INFORMATION_MODE = "retrospective_archive_ingestion_cutoff_proxy_strictly_earlier_fitted_labels" as const;
export const HOUSE_POLLING_POLICY_SHA = "0fe3dca9f2a2eb1bb2098de4bfc73e0d4f634553805c15c4eae874c5fa25cdac";
export const HOUSE_POLLING_AUDIT_SHA = "86e537ae80784dd937e4c90dc6655afe3f0de1b6cb737ce3c5bd89011282b1e9";
export const HOUSE_POLLING_ADMISSION_SHA = "2407128a0e8a5853ac55422be3726f6293612cb553710991b0f427b0de69bf76";
export const HOUSE_POLLING_CATALOG_SHA = "9b35f2a63c895cf6b01de9ac3542b2f7abcb68a555aabdc73fedcb5d55683b0c";
export type HouseGroupBloc = "D" | "R" | "O";
export interface HouseGroupProbabilities { D: number; R: number; O: number }
export interface HouseGroupMember { id: string; name: string; kind: "named_candidate" | "source_reported_name" | "pooled_unidentified"; sourceDisplayName?:string }
export interface HouseResultGroup {
  id: HouseGroupBloc;
  label: string;
  members: HouseGroupMember[];
  namedSingletonResolutionEligible: boolean;
  incumbentGroup: boolean;
  winProbability: number;
  individualCandidateProbability: number | null;
}
export type HouseGroupObserved =
  | { status: "observed"; groupId: HouseGroupBloc; winnerNames: string[]; memberId: string | null; sourceBasis: string }
  | { status: "unavailable"; reason: "annulled_election_no_certified_target" };
export interface HousePollingGroupRace {
  id: string;
  districtKey: string;
  cycle: number;
  ruleFamily: "plurality" | "ranked_choice" | "majority_runoff";
  groups: HouseResultGroup[];
  probabilities: HouseGroupProbabilities;
  conditionalProbabilityMeans: HouseGroupProbabilities;
  structural: boolean;
  probabilityMethod: "modeled_joint_draw_frequency" | "structural_conditional_on_source_support";
  winnerMethod: "categorical_ultimate_group" | "fundamentals_plus_polls_pair_with_categorical_outside_mass";
  pollCount: number;
  pollingMethod: "direct_D_R_poll_likelihood" | "shared_national_factor_only";
  nominalCutoff: string;
  comparisonKey: string;
  supportMode: "final_roster_retrospective_proxy";
  observed: HouseGroupObserved;
  groupScoreEligible: boolean;
  voteShareReferenceAvailable: boolean;
}
export type HouseGroupScoreStratum = "all_scored" | "legacy_subset" | "newly_included" | "competitive" | "newly_included_competitive";
export interface HousePollingGroupRaceScore {
  stratum: HouseGroupScoreStratum;
  n: number;
  categoricalBrier: number;
  binaryDBrier: number;
  binaryRBrier: number;
  logScoreWithDeclaredFloor: number;
  logScoreFloor: number;
  zeroEmpiricalTrueOutcomes: number;
}
export interface HousePollingGroupResult {
  id: string;
  cycle: number;
  configuration: string;
  targetId: "house_aggregate_polling_group_winner_completion_v1";
  reviewStatus: "reviewed_experimental_research";
  comparisonFamily: "house_ultimate_nominee_group_winner_v1";
  sourceMode: "retrospective_support_proxy";
  reviewScope: string;
  approximations: string[];
  provenance: { predictionSha256: string; sourcePanelSha256: string; auditSha256: string; sourcePolicySha256: string };
  /** Identity only; legacy shares require a separately verified exact legacy publication. */
  legacyRunId: string;
  races: HousePollingGroupRace[];
  raceScores: HousePollingGroupRaceScore[];
  raceDiagnostics: { maxGroupWinMcse: number; maxGroupWinRhat: number; conditionalOnSavedBanks: true; seatMeanPrecisionPassed: false };
  polling: true;
  informationMode: typeof HOUSE_POLLING_INFORMATION_MODE;
  pollingCoverage: HousePollingCoverage;
  conditionalShareRunId: string;
  conditionalShareSha256: string;
  nativeMetrics: Record<string, unknown>;
  publicChamberControl: false;
  publicTipping: false;
}
export interface HousePollingCoverage { totalDistricts: 435; shareEligibleDistricts: number; polledDistricts: number; polls: number; horizonDays: 14 }
const BLOCS: HouseGroupBloc[] = ["D", "R", "O"];
const MODELS = ["house_history_refreshed_v1", "house_presidential_slope_v1", "house_relative_lean_v1", "house_relative_context_v1"];
const CURRENT_SEATS: Record<string, number> = { AL:7,AK:1,AZ:9,AR:4,CA:52,CO:8,CT:5,DE:1,FL:28,GA:14,HI:2,ID:2,IL:17,IN:9,IA:4,KS:4,KY:6,LA:6,ME:2,MD:8,MA:9,MI:13,MN:8,MS:4,MO:8,MT:2,NE:3,NV:4,NH:2,NJ:12,NM:3,NY:26,NC:14,ND:1,OH:15,OK:5,OR:6,PA:17,RI:2,SC:7,SD:1,TN:9,TX:38,UT:4,VT:1,VA:11,WA:10,WV:2,WI:8,WY:1 };
const record = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const text = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0 && v.length < 4000 && !/[\u0000-\u001f\u007f]/.test(v);
const strings = (v: unknown): v is string[] => Array.isArray(v) && v.every(text);
const fraction = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1;
const hash = (v: unknown): v is string => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);
const pin = (v: unknown): v is { sha256: string } => record(v) && hash(v.sha256);
const near = (a: number, b: number) => Math.abs(a - b) <= 1e-10;
function strictJson(value:unknown):string {
  const normalize=(v:unknown):unknown=>{
    if(Array.isArray(v))return v.map(normalize);
    if(record(v))return Object.fromEntries(Object.keys(v).sort().map(key=>[key,normalize(v[key])]));
    if(v===undefined||(typeof v==="number"&&!Number.isFinite(v)))throw Error("Invalid normalized value.");
    return v;
  };
  return JSON.stringify(normalize(value));
}
function distribution(v: unknown): HouseGroupProbabilities | null {
  if (!record(v) || Object.keys(v).length !== 3 || !BLOCS.every(b => fraction(v[b]))) return null;
  const p = { D: v.D as number, R: v.R as number, O: v.O as number };
  return Math.abs(p.D + p.R + p.O - 1) <= 1e-12 ? p : null;
}
function districtKeys(cycle: number): Set<string> {
  const seats = cycle < 2022 ? { ...CURRENT_SEATS, CA:53,CO:7,FL:27,IL:18,MI:14,MT:1,NY:27,NC:13,OH:16,OR:5,PA:18,TX:36,WV:3 } : CURRENT_SEATS;
  return new Set(Object.entries(seats).flatMap(([state, count]) => Array.from({ length: count }, (_, i) => `${state}-${String(count === 1 ? 0 : i + 1).padStart(2, "0")}`)));
}
function summarizedScore(stratum: HouseGroupScoreStratum, selected: HousePollingGroupRace[]): HousePollingGroupRaceScore {
  let categorical = 0, d = 0, r = 0, log = 0, zeros = 0;
  const floor = 1e-6; // Exact declared floor in the separately reviewed polling scores.py.
  for (const race of selected) {
    if (race.observed.status !== "observed") throw Error("Missing score truth");
    const truth = race.observed.groupId;
    for (const bloc of BLOCS) { const error = (race.probabilities[bloc] - Number(bloc === truth)) ** 2;
      categorical += error; if (bloc === "D") d += error; if (bloc === "R") r += error; }
    const probability = race.probabilities[truth]; log -= Math.log(Math.max(probability, floor)); if (!probability) zeros++;
  }
  return {stratum,n:selected.length,categoricalBrier:categorical/selected.length,binaryDBrier:d/selected.length,
    binaryRBrier:r/selected.length,logScoreWithDeclaredFloor:log/selected.length,logScoreFloor:floor,zeroEmpiricalTrueOutcomes:zeros};
}
function parseScores(value: unknown, races: HousePollingGroupRace[], cycle: number): HousePollingGroupRaceScore[] | null {
  if (!record(value) || value.cycle !== cycle || value.tier !== ([2018,2022].includes(cycle) ? "A" : "B") ||
    value.forecast_races !== 435 || value.scored_races !== races.filter(r=>r.groupScoreEligible).length ||
    !Array.isArray(value.race_scores) || value.race_scores.length !== 4 || !Array.isArray(value.share_scores) ||
    !record(value.diagnostics) || value.diagnostics.group_win_passed !== true ||
    !fraction(value.diagnostics.max_group_win_mcse) || value.diagnostics.max_group_win_mcse > .005 ||
    typeof value.diagnostics.max_group_win_rhat !== "number" || !Number.isFinite(value.diagnostics.max_group_win_rhat) ||
    value.diagnostics.max_group_win_rhat < .9 || value.diagnostics.max_group_win_rhat > 1.01 ||
    value.diagnostics.seat_mean_precision_passed !== false || value.diagnostics.conditional_on_saved_banks !== true) return null;
  const masks: Record<string,(r:HousePollingGroupRace)=>boolean> = {
    all435_groups:()=>true,polled_groups:r=>r.pollCount>0,unpolled_groups:r=>r.pollCount===0,noncore_groups:r=>!r.voteShareReferenceAvailable,
  };
  const seen = new Set<string>();
  for (const score of value.race_scores) {
    if (!record(score) || typeof score.subset !== "string" || !masks[score.subset] || seen.has(score.subset) ||
      score.cycle !== cycle || score.variant !== "primary" || score.race_numerical_passed !== true || score.forecast_races !== 435) return null;
    seen.add(score.subset);
    const selected = races.filter(r=>r.groupScoreEligible && masks[score.subset as string](r));
    const calculated = summarizedScore("all_scored",selected);
    const accuracy = selected.filter(r=>r.observed.status === "observed" && r.probabilities[r.observed.groupId] === Math.max(...Object.values(r.probabilities))).length/selected.length;
    if (score.n !== selected.length || !near(score.categorical_brier as number,calculated.categoricalBrier) ||
      !near(score.binary_D_brier as number,calculated.binaryDBrier) || !near(score.log_loss as number,calculated.logScoreWithDeclaredFloor) ||
      !near(score.accuracy as number,accuracy)) return null;
  }
  return (["all_scored","legacy_subset","newly_included"] as const).map(name=>summarizedScore(name,races.filter(r=>r.groupScoreEligible &&
    (name==="all_scored" || r.voteShareReferenceAvailable === (name==="legacy_subset")))));
}
/** Structural validation only. The publication loader must separately verify sealed hashes and qualification. */
export interface HousePollingGroupResultBinding { sourcePolicySha256: string }
export function parseHousePollingGroupResult(value: unknown, binding: HousePollingGroupResultBinding): HousePollingGroupResult | null {
  try {
    if (!binding || !hash(binding.sourcePolicySha256) || !record(value) || value.schema !== "HouseFullRosterPollingReviewedResultV1" || value.review_status !== "reviewed_experimental_research" || value.target_id !== "house_aggregate_polling_group_winner_completion_v1" || ![2018,2020,2022,2024].includes(value.cycle as number) || !text(value.configuration) || !text(value.id) || value.race_count !== 435 || !Array.isArray(value.races) || value.races.length !== 435 || value.probabilities_unit !== "unit_interval_probability_mass" || value.population_count_stack !== false || value.current2026 !== false || value.district_geometry !== false || value.polling !== true || value.full_roster_group_forecasts !== true || value.public_chamber_control_qualified !== false || value.source_mode !== "retrospective_support_proxy" || !strings(value.approximations) || !value.approximations.length || !text(value.review_scope) || value.source_policy !== "source-capability-policy-v1.json" || value.comparison_family !== "house_ultimate_nominee_group_winner_v1" || value.no_within_O_candidate_allocation !== true || !pin(value.prediction) || !pin(value.source_panel) || !pin(value.audit)) return null;
    if (binding.sourcePolicySha256 !== HOUSE_POLLING_POLICY_SHA || !pin(value.source_policy_pin) ||
      value.source_policy_pin.sha256 !== HOUSE_POLLING_POLICY_SHA || value.audit.sha256 !== HOUSE_POLLING_AUDIT_SHA ||
      !pin(value.conditional_share_packet) || !record(value.polling_coverage) || !pin(value.source_admission) || value.source_admission.sha256 !== HOUSE_POLLING_ADMISSION_SHA) return null;
    if (value.public_tipping_qualified === true || value.tipping === true) return null;
    const cycle = value.cycle as number, model = MODELS.find(m => value.configuration === `${m}_polling_v1`);
    if (!model || value.id !== `${model}-${cycle}-h14-full435-polls-v1` || value.legacy_share_packet !== `research/house-benchmark-20261020/frontend-results-v2/${model}-${cycle}-h14.json`) return null;
    const expected = districtKeys(cycle), seen = new Set<string>(), races: HousePollingGroupRace[] = [];
    for (const r of value.races) {
      if (!record(r) || typeof r.district_key !== "string" || !expected.has(r.district_key) || seen.has(r.district_key) || r.cycle !== cycle || r.race_id !== `${cycle}:${r.district_key}:house:regular` || !["plurality","ranked_choice","majority_runoff"].includes(r.rule_family as string) || !Array.isArray(r.groups) || !r.groups.length || r.groups.length > 3 || r.within_group_model !== null || r.exact_electoral_rule_simulation !== false || r.prediction_status !== "reviewed_experimental_research" || r.probability_unit !== "unit_interval_probability_mass" || r.comparison_key !== `house_group_winner_435:${cycle}:h14:panel-v3` || r.support_mode !== "final_roster_retrospective_proxy" || r.nominal_cutoff !== ({2018:"2018-10-23",2020:"2020-10-20",2022:"2022-10-25",2024:"2024-10-22"} as Record<number,string>)[cycle] || typeof r.source_conditional_structural !== "boolean" || typeof r.group_score_eligible !== "boolean" || typeof r.vote_share_reference_available !== "boolean") return null;
      const probabilities = distribution(r.group_win_probabilities), conditional = distribution(r.conditional_probability_means);
      if (!probabilities || !conditional) return null;
      seen.add(r.district_key);
      const groups: HouseResultGroup[] = [], ids = new Set<HouseGroupBloc>(), memberIds = new Set<string>();
      for (const g of r.groups) {
        if (!record(g) || !BLOCS.includes(g.id as HouseGroupBloc) || ids.has(g.id as HouseGroupBloc) || !text(g.label) || !Array.isArray(g.members) || !g.members.length || typeof g.named_singleton_resolution_eligible !== "boolean" || typeof g.incumbent_group !== "boolean") return null;
        const id = g.id as HouseGroupBloc, members: HouseGroupMember[] = [];
        ids.add(id);
        for (const m of g.members) {
          if (!record(m) || !text(m.id) || memberIds.has(m.id) || !text(m.name) || !(id === "O" ? ["source_reported_name","pooled_unidentified"] : ["named_candidate","pooled_unidentified"]).includes(m.kind as string)) return null;
          if (m.sourceDisplayName !== undefined && !text(m.sourceDisplayName)) return null;
          memberIds.add(m.id); members.push({ id: m.id, name: m.name, kind: m.kind as HouseGroupMember["kind"],
            ...(typeof m.sourceDisplayName === "string" ? {sourceDisplayName:m.sourceDisplayName} : {}) });
        }
        const eligible = id !== "O" && members.length === 1 && members[0].kind === "named_candidate";
        if (g.named_singleton_resolution_eligible !== eligible || g.win_probability !== probabilities[id] || g.individual_candidate_probability !== (eligible ? probabilities[id] : null)) return null;
        groups.push({ id, label:g.label, members, namedSingletonResolutionEligible:eligible, incumbentGroup:g.incumbent_group, winProbability:probabilities[id], individualCandidateProbability:eligible ? probabilities[id] : null });
      }
      if (BLOCS.some(b => !ids.has(b) && (probabilities[b] !== 0 || conditional[b] !== 0))) return null;
      const structural = r.source_conditional_structural;
      if (structural !== (groups.length === 1) || r.probability_method !== (structural ? "structural_conditional_on_source_support" : "modeled_joint_draw_frequency") || (structural && (probabilities[groups[0].id] !== 1 || conditional[groups[0].id] !== 1))) return null;
      if (!Number.isInteger(r.poll_count) || (r.poll_count as number) < 0 ||
        r.polling_method !== ((r.poll_count as number)>0 ? "direct_D_R_poll_likelihood" : "shared_national_factor_only") ||
        (!r.vote_share_reference_available && r.poll_count !== 0)) return null;
      const expectedMethod = r.vote_share_reference_available ? "fundamentals_plus_polls_pair_with_categorical_outside_mass" : "categorical_ultimate_group";
      if (r.winner_method !== expectedMethod) return null;
      const t = r.observed_result;
      if (!record(t) || !strings(t.winner_names)) return null;
      let observed: HouseGroupObserved;
      const missing = cycle === 2018 && r.district_key === "NC-09";
      if (missing) {
        if (t.status !== "unavailable" || t.group_id !== null || t.winner_names.length !== 0 || t.member_id !== null || t.source_basis !== null || t.unavailable_reason !== "annulled_election_no_certified_target" || r.group_score_eligible !== false) return null;
        observed = { status:"unavailable", reason:"annulled_election_no_certified_target" };
      } else {
        if (t.status !== "observed" || !ids.has(t.group_id as HouseGroupBloc) || t.winner_names.length !== 1 || !text(t.source_basis) || t.unavailable_reason !== null || r.group_score_eligible !== true || (t.member_id !== null && (!text(t.member_id) || !groups.find(g => g.id === t.group_id)!.members.some(m => m.id === t.member_id)))) return null;
        // Validate an explicit member identity within the observed group; names never establish identity.
        observed = { status:"observed",groupId:t.group_id as HouseGroupBloc,winnerNames:[...t.winner_names],memberId:t.member_id as string|null,sourceBasis:t.source_basis };
      }
      races.push({ pollCount:r.poll_count as number,pollingMethod:r.polling_method as HousePollingGroupRace["pollingMethod"],id:r.race_id as string,districtKey:r.district_key,cycle,ruleFamily:r.rule_family as HousePollingGroupRace["ruleFamily"],groups,probabilities,conditionalProbabilityMeans:conditional,structural,probabilityMethod:r.probability_method as HousePollingGroupRace["probabilityMethod"],winnerMethod:expectedMethod,nominalCutoff:r.nominal_cutoff as string,comparisonKey:r.comparison_key as string,supportMode:"final_roster_retrospective_proxy",observed,groupScoreEligible:r.group_score_eligible,voteShareReferenceAvailable:r.vote_share_reference_available });
    }
    const raceScores = parseScores(value.metrics,races,cycle);
    if (!raceScores) return null;
    const coverage = value.polling_coverage;
    if (coverage.total_districts !== 435 || coverage.horizon_days !== 14 ||
      coverage.share_eligible_districts !== races.filter(r=>r.voteShareReferenceAvailable).length ||
      coverage.polled_districts !== races.filter(r=>r.pollCount>0).length ||
      coverage.polls !== races.reduce((n,r)=>n+r.pollCount,0)) return null;
    return { polling:true,informationMode:HOUSE_POLLING_INFORMATION_MODE,
      pollingCoverage:{totalDistricts:435,shareEligibleDistricts:coverage.share_eligible_districts as number,
        polledDistricts:coverage.polled_districts as number,polls:coverage.polls as number,horizonDays:14},
      conditionalShareRunId:`${model}-${cycle}-h14-polls-v1`,conditionalShareSha256:value.conditional_share_packet.sha256,
      nativeMetrics:value.metrics as Record<string,unknown>,id:value.id,cycle,configuration:value.configuration,targetId:"house_aggregate_polling_group_winner_completion_v1",reviewStatus:"reviewed_experimental_research",comparisonFamily:"house_ultimate_nominee_group_winner_v1",sourceMode:"retrospective_support_proxy",reviewScope:value.review_scope,approximations:[...value.approximations],provenance:{predictionSha256:value.prediction.sha256,sourcePanelSha256:value.source_panel.sha256,auditSha256:value.audit.sha256,sourcePolicySha256:binding.sourcePolicySha256},legacyRunId:`${model}-${cycle}-h14`,races,raceScores,raceDiagnostics:{maxGroupWinMcse:(value.metrics as { diagnostics: { max_group_win_mcse: number } }).diagnostics.max_group_win_mcse,maxGroupWinRhat:(value.metrics as { diagnostics: { max_group_win_rhat: number } }).diagnostics.max_group_win_rhat,conditionalOnSavedBanks:true,seatMeanPrecisionPassed:false},publicChamberControl:false,publicTipping:false };
  } catch { return null; }
}

/** Revalidate a path-free FP API view through its separate native contract. */
export function parseHousePollingGroupView(value: unknown): HousePollingGroupResult | null {
  try {
    if (!record(value) || value.polling !== true || value.informationMode !== HOUSE_POLLING_INFORMATION_MODE || value.targetId !== "house_aggregate_polling_group_winner_completion_v1" || value.reviewStatus !== "reviewed_experimental_research" || value.comparisonFamily !== "house_ultimate_nominee_group_winner_v1" || value.sourceMode !== "retrospective_support_proxy" || value.publicChamberControl !== false || value.publicTipping !== false || !record(value.provenance) || ![value.provenance.predictionSha256,value.provenance.sourcePanelSha256,value.provenance.auditSha256,value.provenance.sourcePolicySha256].every(hash) || !Array.isArray(value.races) || !Array.isArray(value.raceScores) || !record(value.raceDiagnostics) || value.raceDiagnostics.conditionalOnSavedBanks !== true || value.raceDiagnostics.seatMeanPrecisionPassed !== false) return null;
    const races = value.races.map(r => {
      if (!record(r) || !Array.isArray(r.groups) || !record(r.observed)) throw Error("Invalid race view");
      return { poll_count:r.pollCount,polling_method:r.pollingMethod,race_id:r.id,district_key:r.districtKey,cycle:r.cycle,rule_family:r.ruleFamily,groups:r.groups.map(g => {
        if (!record(g)) throw Error("Invalid group view");
        return { id:g.id,label:g.label,members:g.members,named_singleton_resolution_eligible:g.namedSingletonResolutionEligible,incumbent_group:g.incumbentGroup,win_probability:g.winProbability,individual_candidate_probability:g.individualCandidateProbability };
      }),group_win_probabilities:r.probabilities,conditional_probability_means:r.conditionalProbabilityMeans,source_conditional_structural:r.structural,probability_method:r.probabilityMethod,winner_method:r.winnerMethod,nominal_cutoff:r.nominalCutoff,comparison_key:r.comparisonKey,support_mode:r.supportMode,within_group_model:null,exact_electoral_rule_simulation:false,prediction_status:"reviewed_experimental_research",probability_unit:"unit_interval_probability_mass",group_score_eligible:r.groupScoreEligible,vote_share_reference_available:r.voteShareReferenceAvailable,observed_result:r.observed.status === "observed" ? { status:r.observed.status,group_id:r.observed.groupId,winner_names:r.observed.winnerNames,member_id:r.observed.memberId,source_basis:r.observed.sourceBasis,unavailable_reason:null } : { status:r.observed.status,group_id:null,winner_names:[],member_id:null,source_basis:null,unavailable_reason:r.observed.reason } };
    });
    const parsed = parseHousePollingGroupResult({ schema:"HouseFullRosterPollingReviewedResultV1",id:value.id,cycle:value.cycle,configuration:value.configuration,target_id:value.targetId,review_status:value.reviewStatus,race_count:races.length,races,probabilities_unit:"unit_interval_probability_mass",population_count_stack:false,current2026:false,district_geometry:false,polling:true,full_roster_group_forecasts:true,public_chamber_control_qualified:value.publicChamberControl,source_mode:value.sourceMode,approximations:value.approximations,review_scope:value.reviewScope,source_policy:"source-capability-policy-v1.json",comparison_family:value.comparisonFamily,no_within_O_candidate_allocation:true,prediction:{sha256:value.provenance.predictionSha256},source_panel:{sha256:value.provenance.sourcePanelSha256},audit:{sha256:value.provenance.auditSha256},legacy_share_packet:`research/house-benchmark-20261020/frontend-results-v2/${value.legacyRunId}.json`,source_admission:{sha256:HOUSE_POLLING_ADMISSION_SHA},source_policy_pin:{sha256:value.provenance.sourcePolicySha256},conditional_share_packet:{sha256:value.conditionalShareSha256},
      polling_coverage:record(value.pollingCoverage)?{total_districts:value.pollingCoverage.totalDistricts,share_eligible_districts:value.pollingCoverage.shareEligibleDistricts,polled_districts:value.pollingCoverage.polledDistricts,polls:value.pollingCoverage.polls,horizon_days:value.pollingCoverage.horizonDays}:null,
      metrics:value.nativeMetrics },{sourcePolicySha256:value.provenance.sourcePolicySha256 as string});
    // Duplicated summaries must agree with the native source-derived values;
    // silently repairing a contradictory API response would hide tampering.
    return parsed && strictJson(parsed) === strictJson(value) ? parsed : null;
  } catch { return null; }
}
