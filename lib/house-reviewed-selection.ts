import type { HouseReviewedGroupResult as HouseGroupResult, HouseReviewedCatalog } from "./house-reviewed-catalog.ts";
import { groupRunId } from "./house-reviewed-catalog.ts";
/** URLs select exact saved results; no component refitting or cross-year district transfer. */
export function groupSelectionUrl(result: HouseGroupResult, raceId?: string | null, view?: string | null, comparisonId?: string | null): string {
  const params = new URLSearchParams({ chamber:"house",run:result.id });
  if (raceId && result.races.some(race=>race.id===raceId)) params.set("race",raceId);
  if (view === "actual") params.set("view","actual");
  if (comparisonId && comparisonId !== result.id && groupRunId(comparisonId) && comparisonId.includes(`-${result.cycle}-h14-full435-`)) params.set("compare",comparisonId);
  return `/?${params}`;
}
export function groupSelectionForResult(result: HouseGroupResult, previous: HouseGroupResult | null, search: string): string {
  const params = new URLSearchParams(search);
  const sameElection = !previous || (previous.cycle===result.cycle && previous.comparisonFamily===result.comparisonFamily && previous.provenance.sourcePanelSha256===result.provenance.sourcePanelSha256);
  return groupSelectionUrl(result,sameElection?params.get("race"):null,params.get("view"),sameElection?params.get("compare"):null);
}

/** Exact saved year/model/measurement selection; never substitute a different model or state. */
export function reviewedGroupSelection(catalog:HouseReviewedCatalog,cycle:number,fundamentalsId:string,polling:boolean):string|null {
  return catalog.runs.find(r=>r.cycle===cycle&&r.fundamentalsId===fundamentalsId&&r.polling===polling)?.id??null;
}
