import { canonicalGroupJson, parseHouseGroupPublication, type HouseGroupPublication } from "./house-group-catalog.ts";
export type HouseGroupComparisonVerification = { status:"comparable"; reason?:never } | { status:"incomparable"; reason:string };
/** Descriptive comparison of exact same-election group targets, without paired uncertainty. */
export async function verifyHouseGroupComparison(primary: HouseGroupPublication, other: HouseGroupPublication): Promise<HouseGroupComparisonVerification> {
  const unavailable = (reason:string):HouseGroupComparisonVerification=>({status:"incomparable",reason});
  try {
    const [a,b] = await Promise.all([parseHouseGroupPublication(primary),parseHouseGroupPublication(other)]);
    if (!a || !b) return unavailable("An exact saved group result or its geography could not be verified.");
    if (a.result.id===b.result.id) return unavailable("Choose two different saved group models.");
    if (a.result.cycle!==b.result.cycle || a.result.targetId!==b.result.targetId || a.result.comparisonFamily!==b.result.comparisonFamily || a.result.provenance.sourcePanelSha256!==b.result.provenance.sourcePanelSha256 || a.result.provenance.sourcePolicySha256!==b.result.provenance.sourcePolicySha256 || a.catalog.qualificationSha256!==b.catalog.qualificationSha256 || a.entry.mapVintage!==b.entry.mapVintage || a.entry.geographySha256!==b.entry.geographySha256 || a.entry.geographyContentSha256!==b.entry.geographyContentSha256) return unavailable("Group comparisons require the same election, map, reviewed source panel and outcome target.");
    const support = (publication:HouseGroupPublication)=>publication.result.races.map(r=>({id:r.id,districtKey:r.districtKey,comparisonKey:r.comparisonKey,ruleFamily:r.ruleFamily,supportMode:r.supportMode,nominalCutoff:r.nominalCutoff,structural:r.structural,winnerMethod:r.winnerMethod,groupScoreEligible:r.groupScoreEligible,voteShareReferenceAvailable:r.voteShareReferenceAvailable,observed:r.observed,groups:r.groups.map(g=>({id:g.id,label:g.label,members:[...g.members].sort((x,y)=>x.id.localeCompare(y.id)),namedSingletonResolutionEligible:g.namedSingletonResolutionEligible,incumbentGroup:g.incumbentGroup})).sort((x,y)=>x.id.localeCompare(y.id))})).sort((x,y)=>x.id.localeCompare(y.id));
    if (canonicalGroupJson(support(a))!==canonicalGroupJson(support(b))) return unavailable("The saved results differ in district support, candidate groups, observed outcomes or scoring masks.");
    if (canonicalGroupJson(a.result.raceScores.map(s=>({stratum:s.stratum,n:s.n,logScoreFloor:s.logScoreFloor})))!==canonicalGroupJson(b.result.raceScores.map(s=>({stratum:s.stratum,n:s.n,logScoreFloor:s.logScoreFloor})))) return unavailable("The saved score summaries use different targets or scoring coverage.");
    return {status:"comparable"};
  } catch { return unavailable("The exact group comparison could not be verified."); }
}
