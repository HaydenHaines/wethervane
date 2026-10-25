import { reviewedGroupRunId, parseHouseReviewedPublication, houseReviewedCatalog, canonicalGroupJson, reviewedDigest,
  type HouseReviewedPublication, type HouseReviewedCatalogEntry } from "./house-reviewed-catalog.ts";
import {parseHousePollingShares,type HousePollingShareResult} from "./house-polling-shares.ts";
export async function loadHouseReviewedGroup(id:string,signal:AbortSignal):Promise<HouseReviewedPublication> {
  if (!reviewedGroupRunId(id)) throw Error("This reviewed group result is not published.");
  const response = await fetch(`/api/chambers/house/groups?${new URLSearchParams({run:id})}`,{signal,cache:"no-store"});
  if (!response.ok) throw Error("The reviewed House result could not be loaded.");
  const parsed = await parseHouseReviewedPublication(await response.json());
  if (!parsed || parsed.result.id!==id) throw Error("Returned House result does not match the requested selection.");
  return parsed;
}
export {loadHouseReviewedGroup as loadHouseGroup};
export async function loadHousePollingShares(entry:HouseReviewedCatalogEntry,signal:AbortSignal):Promise<HousePollingShareResult> {
  const exact = houseReviewedCatalog().runs.find(r=>r.id===entry.id);
  if (!exact?.polling || canonicalGroupJson(exact)!==canonicalGroupJson(entry)) throw Error("The exact polling-share binding is unavailable.");
  const response = await fetch(`/api/chambers/house/groups/shares?${new URLSearchParams({run:entry.conditionalShareRunId})}`,
    {signal,cache:"no-store"});
  if(!response.ok)throw Error("The conditional polling-share result could not be loaded.");
  const value=await response.json();
  const result=parseHousePollingShares(value.result);
  if(value.schemaVersion!==1||value.status!=="available"||value.family!=="house_polling_shares"||
    canonicalGroupJson(value.entry)!==canonicalGroupJson(entry)||value.resultSha256!==entry.conditionalShareSha256||
    !result||result.run.id!==entry.conditionalShareRunId||await reviewedDigest(result)!==entry.conditionalShareContentSha256)
    throw Error("The conditional shares differ from their exact reviewed result.");
  return result;
}
