import type { HouseGroupResult } from "./house-group-result.ts";
import { groupRunId } from "./house-group-catalog.ts";
/** URLs select exact saved results; no component refitting or cross-year district transfer. */
export function groupSelectionUrl(result: HouseGroupResult, raceId?: string | null, view?: string | null, comparisonId?: string | null): string {
  const params = new URLSearchParams({ chamber:"house",run:result.id });
  if (raceId && result.races.some(race=>race.id===raceId)) params.set("race",raceId);
  if (view === "actual") params.set("view","actual");
  if (comparisonId && comparisonId !== result.id && groupRunId(comparisonId) && comparisonId.endsWith(`-${result.cycle}-h14-full435-v1`)) params.set("compare",comparisonId);
  return `/?${params}`;
}
export function groupSelectionForResult(result: HouseGroupResult, previous: HouseGroupResult | null, search: string): string {
  const params = new URLSearchParams(search);
  const sameElection = !previous || (previous.cycle===result.cycle && previous.targetId===result.targetId && previous.comparisonFamily===result.comparisonFamily && previous.provenance.sourcePanelSha256===result.provenance.sourcePanelSha256);
  return groupSelectionUrl(result,sameElection?params.get("race"):null,params.get("view"),sameElection?params.get("compare"):null);
}
