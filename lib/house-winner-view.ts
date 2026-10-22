/** Invented frontend presentation fixtures only, not an owner publication schema. */
export interface HouseWinnerGroup {
  id: string;
  bloc: "D" | "R" | "O";
  namedSingletonResolutionEligible?: boolean;
  members: { id: string; label: string; kind: "named_person" | "pooled_or_residual" }[];
}
export type HouseWinnerPrediction =
  | { status: "modeled"; probabilities: { groupId: string; probability: number }[] }
  | { status: "structural"; groupId: string; reason: string }
  | { status: "unavailable"; reason: string };
export type HouseWinnerTruth =
  | { status: "observed"; groupId: string; sourceBasis: string; winnerNames?: string[] }
  | { status: "unavailable"; reason: string };
export interface HouseWinnerFixture {
  fixtureOnly: true;
  sourceStatus: "invented_fixture";
  district: string;
  stage: "ultimate_outcome";
  resolution: "party_group";
  ruleFamily: "plurality" | "ranked_choice" | "majority_runoff";
  approximationNote: string;
  groups: HouseWinnerGroup[];
  prediction: HouseWinnerPrediction;
  truth: HouseWinnerTruth;
}
export interface HouseResolvedCandidate {
  id: string;
  label: string;
  probability: number;
  basis: "modeled_group_probability" | "structural_group_support";
}
export interface HouseWinnerView extends Omit<HouseWinnerFixture, "groups"> {
  groups: (HouseWinnerGroup & { resolvedCandidate?: HouseResolvedCandidate })[];
}
const text = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;
const record = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const probability = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1;
/** Never infer prediction admission from roster shape, member count, names or truth. */
export function houseWinnerView(input: HouseWinnerFixture): HouseWinnerView {
  if (!input || input.fixtureOnly !== true || input.sourceStatus !== "invented_fixture" || input.stage !== "ultimate_outcome" || input.resolution !== "party_group" || !/^[A-Z]{2}-\d{2}$/.test(input.district) || !["plurality", "ranked_choice", "majority_runoff"].includes(input.ruleFamily) || !text(input.approximationNote))
    throw new TypeError("An explicitly invented ultimate party/group presentation fixture is required.");
  const groups = input.groups;
  const memberIds = new Set<string>();
  if (!Array.isArray(groups) || groups.length === 0 || groups.some(g => !record(g) || !text(g.id) || !["D", "R", "O"].includes(g.bloc as string) || !Array.isArray(g.members) || !g.members.length || (g.namedSingletonResolutionEligible !== undefined && typeof g.namedSingletonResolutionEligible !== "boolean") || (g.namedSingletonResolutionEligible === true && (g.members.length !== 1 || g.members[0]?.kind !== "named_person")) || g.members.some(m => {
    if (!record(m) || !text(m.id) || !text(m.label) || !["named_person", "pooled_or_residual"].includes(m.kind as string) || memberIds.has(m.id)) return true;
    memberIds.add(m.id); return false;
  })) || new Set(groups.map(g => g.id)).size !== groups.length || new Set(groups.map(g => g.bloc)).size !== groups.length)
    throw new TypeError("Distinct groups with explicit, disjoint member identities and kinds are required.");
  const ids = new Set(groups.map(g => g.id));
  const p = input.prediction;
  let prediction: HouseWinnerPrediction = { status: "unavailable", reason: "Group prediction is missing or invalid." };
  if (p?.status === "unavailable" && text(p.reason)) prediction = { status: "unavailable", reason: p.reason };
  if (p?.status === "structural" && groups.length === 1 && ids.has(p.groupId) && text(p.reason)) prediction = { status: "structural", groupId: p.groupId, reason: p.reason };
  if (p?.status === "modeled" && Array.isArray(p.probabilities) && p.probabilities.length === groups.length && p.probabilities.every(e => record(e) && ids.has(e.groupId as string) && probability(e.probability)) && new Set(p.probabilities.map(e => e.groupId)).size === groups.length && Math.abs(p.probabilities.reduce((sum, e) => sum + e.probability, 0) - 1) <= 1e-8)
    prediction = { status: "modeled", probabilities: p.probabilities.map(e => ({ groupId: e.groupId, probability: e.probability })) };
  const t = input.truth;
  let truth: HouseWinnerTruth = { status: "unavailable", reason: "Observed ultimate outcome is missing or invalid." };
  if (t?.status === "unavailable" && text(t.reason)) truth = { status: "unavailable", reason: t.reason };
  if (t?.status === "observed" && ids.has(t.groupId) && text(t.sourceBasis) && (t.winnerNames === undefined || (Array.isArray(t.winnerNames) && t.winnerNames.every(text)))) truth = { status: "observed", groupId: t.groupId, sourceBasis: t.sourceBasis, ...(t.winnerNames === undefined ? {} : { winnerNames: [...t.winnerNames] }) };
  // Reconstruct private view fields rather than forwarding undeclared numerical claims.
  return { fixtureOnly: true, sourceStatus: "invented_fixture", district: input.district, stage: "ultimate_outcome", resolution: "party_group", ruleFamily: input.ruleFamily, approximationNote: input.approximationNote, groups: groups.map(g => {
    const member = g.members[0];
    const ready = prediction.status !== "unavailable";
    const resolvedCandidate: HouseResolvedCandidate | undefined = g.namedSingletonResolutionEligible === true && ready
      ? { id: member.id, label: member.label, probability: prediction.status === "modeled" ? prediction.probabilities.find(p => p.groupId === g.id)!.probability : 1, basis: prediction.status === "modeled" ? "modeled_group_probability" : "structural_group_support" }
      : undefined;
    return { id: g.id, bloc: g.bloc, ...(g.namedSingletonResolutionEligible === undefined ? {} : { namedSingletonResolutionEligible: g.namedSingletonResolutionEligible }), members: g.members.map(m => ({ id: m.id, label: m.label, kind: m.kind })), ...(resolvedCandidate ? { resolvedCandidate } : {}) };
  }), prediction, truth };
}
