/** Final experimental group-result adapter. Independent of the frozen D/R v1 contract. */
export type HouseGroupBloc = "D" | "R" | "O";
export interface HouseGroupProbabilities { D: number; R: number; O: number }
export interface HouseGroupMember { id: string; name: string; kind: "named_candidate" | "source_reported_name" | "pooled_unidentified" }
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
export interface HouseGroupRace {
  id: string;
  districtKey: string;
  cycle: number;
  ruleFamily: "plurality" | "ranked_choice" | "majority_runoff";
  groups: HouseResultGroup[];
  probabilities: HouseGroupProbabilities;
  conditionalProbabilityMeans: HouseGroupProbabilities;
  structural: boolean;
  probabilityMethod: "modeled_joint_draw_frequency" | "structural_conditional_on_source_support";
  winnerMethod: "categorical_ultimate_group" | "fundamentals_pair_with_categorical_outside_mass";
  nominalCutoff: string;
  comparisonKey: string;
  supportMode: "final_roster_retrospective_proxy";
  observed: HouseGroupObserved;
  groupScoreEligible: boolean;
  voteShareReferenceAvailable: boolean;
}
export type HouseGroupScoreStratum = "all_scored" | "legacy_subset" | "newly_included" | "competitive" | "newly_included_competitive";
export interface HouseGroupRaceScore {
  stratum: HouseGroupScoreStratum;
  n: number;
  categoricalBrier: number;
  binaryDBrier: number;
  binaryRBrier: number;
  logScoreWithDeclaredFloor: number;
  logScoreFloor: number;
  zeroEmpiricalTrueOutcomes: number;
}
export interface HouseGroupResult {
  id: string;
  cycle: number;
  configuration: string;
  targetId: "house_aggregate_group_winner_completion_v1";
  reviewStatus: "reviewed_experimental_research";
  comparisonFamily: "house_ultimate_nominee_group_winner_v1";
  sourceMode: "retrospective_support_proxy";
  reviewScope: string;
  approximations: string[];
  provenance: { predictionSha256: string; sourcePanelSha256: string; auditSha256: string; sourcePolicySha256: string };
  /** Identity only; legacy shares require a separately verified exact legacy publication. */
  legacyRunId: string;
  races: HouseGroupRace[];
  raceScores: HouseGroupRaceScore[];
  raceDiagnostics: { maxGroupWinMcse: number; maxGroupWinRhat: number; conditionalOnSavedBanks: true; seatMeanPrecisionPassed: false };
  publicChamberControl: false;
  publicTipping: false;
}
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
function distribution(v: unknown): HouseGroupProbabilities | null {
  if (!record(v) || Object.keys(v).length !== 3 || !BLOCS.every(b => fraction(v[b]))) return null;
  const p = { D: v.D as number, R: v.R as number, O: v.O as number };
  return Math.abs(p.D + p.R + p.O - 1) <= 1e-12 ? p : null;
}
function districtKeys(cycle: number): Set<string> {
  const seats = cycle < 2022 ? { ...CURRENT_SEATS, CA:53,CO:7,FL:27,IL:18,MI:14,MT:1,NY:27,NC:13,OH:16,OR:5,PA:18,TX:36,WV:3 } : CURRENT_SEATS;
  return new Set(Object.entries(seats).flatMap(([state, count]) => Array.from({ length: count }, (_, i) => `${state}-${String(count === 1 ? 0 : i + 1).padStart(2, "0")}`)));
}
function parseScores(value: unknown, races: HouseGroupRace[], cycle: number, configuration: string): HouseGroupRaceScore[] | null {
  if (!record(value) || value.cycle !== cycle || value.cell !== configuration || value.loading_factor !== 1 || value.forecast_races !== 435 || value.scored_races !== races.filter(r => r.groupScoreEligible).length || value.tier !== ([2018, 2022].includes(cycle) ? "A" : "B") || !record(value.race_scores) || !record(value.diagnostics) || value.diagnostics.group_win_passed !== true || !fraction(value.diagnostics.max_group_win_mcse) || value.diagnostics.max_group_win_mcse > .005 || typeof value.diagnostics.max_group_win_rhat !== "number" || !Number.isFinite(value.diagnostics.max_group_win_rhat) || value.diagnostics.max_group_win_rhat < .9 || value.diagnostics.max_group_win_rhat > 1.01 || value.diagnostics.seat_mean_precision_passed !== false || value.diagnostics.conditional_on_saved_banks !== true) return null;
  const names: HouseGroupScoreStratum[] = ["all_scored", "legacy_subset", "newly_included", "competitive", "newly_included_competitive"];
  if (Object.keys(value.race_scores).some(k => !names.includes(k as HouseGroupScoreStratum))) return null;
  const scores: HouseGroupRaceScore[] = [];
  for (const name of names) {
    const s = value.race_scores[name];
    if (s === undefined && (name === "competitive" || name === "newly_included_competitive")) continue;
    if (!record(s) || !Number.isInteger(s.n) || (s.n as number) < 1 || (s.n as number) > 435 || typeof s.categorical_brier !== "number" || !Number.isFinite(s.categorical_brier) || s.categorical_brier < 0 || s.categorical_brier > 2 || !fraction(s.binary_D_brier) || !fraction(s.binary_R_brier) || !fraction(s.log_score_floor) || s.log_score_floor <= 0 || typeof s.log_score_with_declared_floor !== "number" || !Number.isFinite(s.log_score_with_declared_floor) || s.log_score_with_declared_floor < 0 || s.log_score_with_declared_floor > -Math.log(s.log_score_floor) + 1e-10 || !Number.isInteger(s.zero_empirical_true_outcomes) || (s.zero_empirical_true_outcomes as number) < 0 || (s.zero_empirical_true_outcomes as number) > (s.n as number) || s.uniform_three_group_categorical_brier !== 2 / 3 || s.binary_D_coinflip_brier !== .25) return null;
    scores.push({ stratum: name, n: s.n as number, categoricalBrier: s.categorical_brier, binaryDBrier: s.binary_D_brier, binaryRBrier: s.binary_R_brier, logScoreWithDeclaredFloor: s.log_score_with_declared_floor, logScoreFloor: s.log_score_floor, zeroEmpiricalTrueOutcomes: s.zero_empirical_true_outcomes as number });
  }
  // Independently check the supplied summaries on the explicit truth and legacy masks.
  for (const name of ["all_scored", "legacy_subset", "newly_included"] as const) {
    const s = scores.find(s => s.stratum === name)!;
    const selected = races.filter(r => r.groupScoreEligible && (name === "all_scored" || r.voteShareReferenceAvailable === (name === "legacy_subset")));
    if (s.n !== selected.length) return null;
    let categorical = 0, d = 0, r = 0, log = 0, zeros = 0;
    for (const race of selected) {
      if (race.observed.status !== "observed") return null;
      const truth = race.observed.groupId;
      for (const b of BLOCS) { const error = (race.probabilities[b] - Number(b === truth)) ** 2; categorical += error; if (b === "D") d += error; if (b === "R") r += error; }
      const p = race.probabilities[truth]; log -= Math.log(Math.max(p, s.logScoreFloor)); if (p === 0) zeros++;
    }
    if (!near(s.categoricalBrier, categorical / s.n) || !near(s.binaryDBrier, d / s.n) || !near(s.binaryRBrier, r / s.n) || !near(s.logScoreWithDeclaredFloor, log / s.n) || s.zeroEmpiricalTrueOutcomes !== zeros) return null;
  }
  const all = scores[0], added = scores.find(s => s.stratum === "newly_included")!, competitive = scores.find(s => s.stratum === "competitive"), newCompetitive = scores.find(s => s.stratum === "newly_included_competitive");
  if (scores.some(s => s.logScoreFloor !== all.logScoreFloor || s.n > all.n) || (newCompetitive && (!competitive || newCompetitive.n > competitive.n || newCompetitive.n > added.n))) return null;
  return scores;
}
/** Structural validation only. The publication loader must separately verify sealed hashes and qualification. */
export interface HouseGroupResultBinding { sourcePolicySha256: string }
export function parseHouseGroupResult(value: unknown, binding: HouseGroupResultBinding): HouseGroupResult | null {
  try {
    if (!binding || !hash(binding.sourcePolicySha256) || !record(value) || value.schema !== "HouseFullRosterReviewedResultV1" || value.review_status !== "reviewed_experimental_research" || value.target_id !== "house_aggregate_group_winner_completion_v1" || ![2018,2020,2022,2024].includes(value.cycle as number) || !text(value.configuration) || !text(value.id) || value.race_count !== 435 || !Array.isArray(value.races) || value.races.length !== 435 || value.probabilities_unit !== "unit_interval_probability_mass" || value.population_count_stack !== false || value.current2026 !== false || value.district_geometry !== false || value.polling !== false || value.full_roster_group_forecasts !== true || value.public_chamber_control_qualified !== false || value.source_mode !== "retrospective_support_proxy" || !strings(value.approximations) || !value.approximations.length || !text(value.review_scope) || value.source_policy !== "source-capability-policy-v1.json" || value.comparison_family !== "house_ultimate_nominee_group_winner_v1" || value.no_within_O_candidate_allocation !== true || !pin(value.prediction) || !pin(value.source_panel) || !pin(value.audit)) return null;
    if (value.public_tipping_qualified === true || value.tipping === true) return null;
    const cycle = value.cycle as number, model = MODELS.find(m => value.configuration === `${m}-loading-1`);
    if (!model || value.id !== `${model}-${cycle}-h14-full435-v1` || value.legacy_share_packet !== `research/house-benchmark-20261020/frontend-results-v2/${model}-${cycle}-h14.json`) return null;
    const expected = districtKeys(cycle), seen = new Set<string>(), races: HouseGroupRace[] = [];
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
          memberIds.add(m.id); members.push({ id: m.id, name: m.name, kind: m.kind as HouseGroupMember["kind"] });
        }
        const eligible = id !== "O" && members.length === 1 && members[0].kind === "named_candidate";
        if (g.named_singleton_resolution_eligible !== eligible || g.win_probability !== probabilities[id] || g.individual_candidate_probability !== (eligible ? probabilities[id] : null)) return null;
        groups.push({ id, label:g.label, members, namedSingletonResolutionEligible:eligible, incumbentGroup:g.incumbent_group, winProbability:probabilities[id], individualCandidateProbability:eligible ? probabilities[id] : null });
      }
      if (BLOCS.some(b => !ids.has(b) && (probabilities[b] !== 0 || conditional[b] !== 0))) return null;
      const structural = r.source_conditional_structural;
      if (structural !== (groups.length === 1) || r.probability_method !== (structural ? "structural_conditional_on_source_support" : "modeled_joint_draw_frequency") || (structural && (probabilities[groups[0].id] !== 1 || conditional[groups[0].id] !== 1))) return null;
      const expectedMethod = r.vote_share_reference_available ? "fundamentals_pair_with_categorical_outside_mass" : "categorical_ultimate_group";
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
      races.push({ id:r.race_id as string,districtKey:r.district_key,cycle,ruleFamily:r.rule_family as HouseGroupRace["ruleFamily"],groups,probabilities,conditionalProbabilityMeans:conditional,structural,probabilityMethod:r.probability_method as HouseGroupRace["probabilityMethod"],winnerMethod:expectedMethod,nominalCutoff:r.nominal_cutoff as string,comparisonKey:r.comparison_key as string,supportMode:"final_roster_retrospective_proxy",observed,groupScoreEligible:r.group_score_eligible,voteShareReferenceAvailable:r.vote_share_reference_available });
    }
    const raceScores = parseScores(value.metrics,races,cycle,value.configuration);
    if (!raceScores) return null;
    return { id:value.id,cycle,configuration:value.configuration,targetId:"house_aggregate_group_winner_completion_v1",reviewStatus:"reviewed_experimental_research",comparisonFamily:"house_ultimate_nominee_group_winner_v1",sourceMode:"retrospective_support_proxy",reviewScope:value.review_scope,approximations:[...value.approximations],provenance:{predictionSha256:value.prediction.sha256,sourcePanelSha256:value.source_panel.sha256,auditSha256:value.audit.sha256,sourcePolicySha256:binding.sourcePolicySha256},legacyRunId:`${model}-${cycle}-h14`,races,raceScores,raceDiagnostics:{maxGroupWinMcse:(value.metrics as { diagnostics: { max_group_win_mcse: number } }).diagnostics.max_group_win_mcse,maxGroupWinRhat:(value.metrics as { diagnostics: { max_group_win_rhat: number } }).diagnostics.max_group_win_rhat,conditionalOnSavedBanks:true,seatMeanPrecisionPassed:false},publicChamberControl:false,publicTipping:false };
  } catch { return null; }
}
/** Source fields useful to importer authors; unknown input must still pass the parser. */
export interface HouseGroupResultSource {
  schema: "HouseFullRosterReviewedResultV1";
  id: string;
  cycle: number;
  configuration: string;
  target_id: "house_aggregate_group_winner_completion_v1";
  review_status: "reviewed_experimental_research";
  races: HouseGroupRaceSource[];
  [key: string]: unknown;
}
export interface HouseGroupRaceSource {
  race_id: string;
  district_key: string;
  cycle: number;
  group_win_probabilities: HouseGroupProbabilities;
  conditional_probability_means: HouseGroupProbabilities;
  groups: {
    id: HouseGroupBloc;
    label: string;
    members: HouseGroupMember[];
    named_singleton_resolution_eligible: boolean;
    incumbent_group: boolean;
    win_probability: number;
    individual_candidate_probability: number | null;
  }[];
  [key: string]: unknown;
}
/** Validate a path-free API view through the same full roster and evidence checks. */
export function parseHouseGroupView(value: unknown): HouseGroupResult | null {
  try {
    if (!record(value) || value.targetId !== "house_aggregate_group_winner_completion_v1" || value.reviewStatus !== "reviewed_experimental_research" || value.comparisonFamily !== "house_ultimate_nominee_group_winner_v1" || value.sourceMode !== "retrospective_support_proxy" || value.publicChamberControl !== false || value.publicTipping !== false || !record(value.provenance) || ![value.provenance.predictionSha256,value.provenance.sourcePanelSha256,value.provenance.auditSha256,value.provenance.sourcePolicySha256].every(hash) || !Array.isArray(value.races) || !Array.isArray(value.raceScores) || !record(value.raceDiagnostics) || value.raceDiagnostics.conditionalOnSavedBanks !== true || value.raceDiagnostics.seatMeanPrecisionPassed !== false) return null;
    const scores: Record<string, unknown> = {};
    for (const s of value.raceScores) {
      if (!record(s) || typeof s.stratum !== "string" || Object.hasOwn(scores,s.stratum)) return null;
      scores[s.stratum] = { n:s.n,categorical_brier:s.categoricalBrier,binary_D_brier:s.binaryDBrier,binary_R_brier:s.binaryRBrier,log_score_with_declared_floor:s.logScoreWithDeclaredFloor,log_score_floor:s.logScoreFloor,zero_empirical_true_outcomes:s.zeroEmpiricalTrueOutcomes,uniform_three_group_categorical_brier:2/3,binary_D_coinflip_brier:.25 };
    }
    const races = value.races.map(r => {
      if (!record(r) || !Array.isArray(r.groups) || !record(r.observed)) throw Error("Invalid race view");
      return { race_id:r.id,district_key:r.districtKey,cycle:r.cycle,rule_family:r.ruleFamily,groups:r.groups.map(g => {
        if (!record(g)) throw Error("Invalid group view");
        return { id:g.id,label:g.label,members:g.members,named_singleton_resolution_eligible:g.namedSingletonResolutionEligible,incumbent_group:g.incumbentGroup,win_probability:g.winProbability,individual_candidate_probability:g.individualCandidateProbability };
      }),group_win_probabilities:r.probabilities,conditional_probability_means:r.conditionalProbabilityMeans,source_conditional_structural:r.structural,probability_method:r.probabilityMethod,winner_method:r.winnerMethod,nominal_cutoff:r.nominalCutoff,comparison_key:r.comparisonKey,support_mode:r.supportMode,within_group_model:null,exact_electoral_rule_simulation:false,prediction_status:"reviewed_experimental_research",probability_unit:"unit_interval_probability_mass",group_score_eligible:r.groupScoreEligible,vote_share_reference_available:r.voteShareReferenceAvailable,observed_result:r.observed.status === "observed" ? { status:r.observed.status,group_id:r.observed.groupId,winner_names:r.observed.winnerNames,member_id:r.observed.memberId,source_basis:r.observed.sourceBasis,unavailable_reason:null } : { status:r.observed.status,group_id:null,winner_names:[],member_id:null,source_basis:null,unavailable_reason:r.observed.reason } };
    });
    return parseHouseGroupResult({ schema:"HouseFullRosterReviewedResultV1",id:value.id,cycle:value.cycle,configuration:value.configuration,target_id:value.targetId,review_status:value.reviewStatus,race_count:races.length,races,probabilities_unit:"unit_interval_probability_mass",population_count_stack:false,current2026:false,district_geometry:false,polling:false,full_roster_group_forecasts:true,public_chamber_control_qualified:value.publicChamberControl,source_mode:value.sourceMode,approximations:value.approximations,review_scope:value.reviewScope,source_policy:"source-capability-policy-v1.json",comparison_family:value.comparisonFamily,no_within_O_candidate_allocation:true,prediction:{sha256:value.provenance.predictionSha256},source_panel:{sha256:value.provenance.sourcePanelSha256},audit:{sha256:value.provenance.auditSha256},legacy_share_packet:`research/house-benchmark-20261020/frontend-results-v2/${value.legacyRunId}.json`,metrics:{cycle:value.cycle,cell:value.configuration,loading_factor:1,tier:[2018,2022].includes(value.cycle as number)?"A":"B",forecast_races:races.length,scored_races:races.filter(r=>r.group_score_eligible===true).length,race_scores:scores,diagnostics:{max_group_win_mcse:value.raceDiagnostics.maxGroupWinMcse,max_group_win_rhat:value.raceDiagnostics.maxGroupWinRhat,group_win_passed:true,seat_mean_precision_passed:value.raceDiagnostics.seatMeanPrecisionPassed,conditional_on_saved_banks:value.raceDiagnostics.conditionalOnSavedBanks}} },{sourcePolicySha256:value.provenance.sourcePolicySha256 as string});
  } catch { return null; }
}
