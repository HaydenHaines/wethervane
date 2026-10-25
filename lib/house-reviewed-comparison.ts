import { canonicalGroupJson, parseHouseReviewedPublication, type HouseReviewedPublication } from "./house-reviewed-catalog.ts";
import { verifyHouseGroupComparison as verifyLegacyComparison } from "./house-group-comparison.ts";
import { parseHouseGroupCatalog, type HouseGroupPublication } from "./house-group-catalog.ts";
export type HouseGroupComparisonVerification = {status:"comparable";reason?:never}|{status:"incomparable";reason:string};
function legacy(publication:HouseReviewedPublication):HouseGroupPublication {
  const runs = publication.catalog.runs.filter(r=>!r.polling).map(({polling,conditionalShareRunId,conditionalShareSha256,
    conditionalShareContentSha256,sourcePolicySha256,sourceCatalogSha256,qualificationSha256,matchedBaselineRunId,...r})=>r);
  const catalog = parseHouseGroupCatalog({...publication.catalog,schemaVersion:1,runs});
  if (!catalog || publication.entry.polling) throw Error("Not a legacy F publication.");
  return {...publication,catalog,entry:catalog.runs.find(r=>r.id===publication.entry.id)!} as HouseGroupPublication;
}
function support(publication:HouseReviewedPublication) {
  const aliases = publication.catalog.aliases;
  return publication.result.races.map(r=>({id:r.id,districtKey:r.districtKey,ruleFamily:r.ruleFamily,structural:r.structural,
    key:r.comparisonKey,cutoff:r.nominalCutoff,mode:r.supportMode,core:r.voteShareReferenceAvailable,scored:r.groupScoreEligible,
    observed:r.observed.status==="observed"?{...r.observed,winnerNames:r.observed.winnerNames.map(name=>
      aliases.find(a=>a.raceId===r.id&&a.party===(r.observed.status==="observed"?r.observed.groupId:"")&&(a.pollingName===name||a.baselineName===name))?.baselineName??name)}:r.observed,
    groups:r.groups.map(g=>({id:g.id,label:g.label,singleton:g.namedSingletonResolutionEligible,incumbent:g.incumbentGroup,
      members:g.members.map(m=>({id:m.id,kind:m.kind,name:aliases.find(a=>a.raceId===r.id&&a.party===g.id&&a.pollingName===m.name)?.baselineName??m.name}))}))}));
}
/** Every comparison score is tied to its explicit outcome IDs, not just a display stratum. */
export function houseGroupScoreMask(publication:HouseReviewedPublication,stratum:string):string[]|null {
  if (!["all_scored","legacy_subset","newly_included"].includes(stratum)) return null;
  return publication.result.races.filter(r=>r.groupScoreEligible&&(stratum==="all_scored"||r.voteShareReferenceAvailable===(stratum==="legacy_subset"))).map(r=>r.id).sort();
}
export function sharedHouseGroupScoreStrata(a:HouseReviewedPublication,b:HouseReviewedPublication):string[] {
  return a.result.raceScores.filter(s=>{const x=houseGroupScoreMask(a,s.stratum),y=houseGroupScoreMask(b,s.stratum);
    return x&&y&&canonicalGroupJson(x)===canonicalGroupJson(y)&&b.result.raceScores.some(t=>t.stratum===s.stratum);}).map(s=>s.stratum);
}
export async function verifyHouseGroupComparison(primary:HouseReviewedPublication,other:HouseReviewedPublication):Promise<HouseGroupComparisonVerification> {
  const unavailable=(reason:string):HouseGroupComparisonVerification=>({status:"incomparable",reason});
  try {
    const [a,b]=await Promise.all([parseHouseReviewedPublication(primary),parseHouseReviewedPublication(other)]);
    if (!a||!b) return unavailable("An exact reviewed result or map could not be verified.");
    if (!a.entry.polling&&!b.entry.polling) return verifyLegacyComparison(legacy(a),legacy(b));
    if(a.result.id===b.result.id||a.result.cycle!==b.result.cycle||a.result.comparisonFamily!==b.result.comparisonFamily||
      a.entry.mapVintage!==b.entry.mapVintage||a.entry.geographySha256!==b.entry.geographySha256||
      a.entry.geographyContentSha256!==b.entry.geographyContentSha256||
      a.result.provenance.sourcePanelSha256!==b.result.provenance.sourcePanelSha256) return unavailable("Comparison requires the same election, map and observable group-winner support.");
    if(a.entry.polling!==b.entry.polling&&!a.catalog.matchedPairs.some(pair=>
      [pair.baselineRunId,pair.pollingRunId].includes(a.result.id)&&[pair.baselineRunId,pair.pollingRunId].includes(b.result.id)))
      return unavailable("This F/FP pair is not one of the independently reviewed matched pairs.");
    if(canonicalGroupJson(support(a))!==canonicalGroupJson(support(b)))return unavailable("Candidate identities, truth, support or scoring masks differ.");
    return {status:"comparable"};
  } catch {return unavailable("The exact reviewed comparison could not be verified.");}
}
