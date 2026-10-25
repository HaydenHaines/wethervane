/** Selectable collection adapter; frozen F validators and publications stay intact. */
import bindings from "../data/house-polling/reviewed-bindings.json" with { type: "json" };
import { canonicalGroupJson, parseHouseGroupCatalog, HOUSE_GROUP_POLICY_SHA, HOUSE_GROUP_SOURCE_CATALOG_SHA,
  HOUSE_GROUP_QUALIFICATION_SHA, type HouseGroupCatalogEntry } from "./house-group-catalog.ts";
import { parseHouseGroupView, type HouseGroupResult, type HouseGroupRace } from "./house-group-result.ts";
import { parseHousePollingGroupView, HOUSE_POLLING_POLICY_SHA, HOUSE_POLLING_CATALOG_SHA,
  HOUSE_POLLING_AUDIT_SHA, type HousePollingGroupResult, type HousePollingGroupRace } from "./house-polling-result.ts";
import { joinDistrictGeography, type DistrictGeography } from "./district-geography.ts";
import { groupRaceForMap } from "./house-group-presentation.ts";

export type HouseReviewedGroupResult = HouseGroupResult | HousePollingGroupResult;
export type HouseReviewedGroupRace = HouseGroupRace | HousePollingGroupRace;
export interface HouseReviewedCatalogEntry extends HouseGroupCatalogEntry {
  polling: boolean;
  conditionalShareRunId: string;
  conditionalShareSha256: string;
  conditionalShareContentSha256: string;
  sourcePolicySha256: string;
  sourceCatalogSha256: string;
  qualificationSha256: string;
  matchedBaselineRunId: string | null;
}
export interface HouseReviewedCatalog {
  schemaVersion: 2;
  qualificationSha256: string;
  sourcePolicySha256: string;
  sourceCatalogSha256: string;
  initialRunId: string;
  runs: HouseReviewedCatalogEntry[];
  matchedPairs: {baselineRunId:string;pollingRunId:string;comparisonKey:string;groupRows:435;shareRows:number}[];
  aliases: {raceId:string;party:string;baselineName:string;pollingName:string;receiptSha256:string}[];
}
export interface HouseReviewedPublication {
  schemaVersion: 1; status: "available"; family: "house_groups";
  catalog: HouseReviewedCatalog; entry: HouseReviewedCatalogEntry;
  result: HouseReviewedGroupResult; geography: DistrictGeography;
}
const record = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
export const reviewedGroupRunId = (v:unknown):v is string => typeof v === "string" &&
  /^house_(history_refreshed|presidential_slope|relative_lean|relative_context)_v1-(2018|2020|2022|2024)-h14-full435-(polls-)?v1$/.test(v);
export const groupRunId = reviewedGroupRunId;
export { canonicalGroupJson };
export async function reviewedDigest(value:unknown):Promise<string> {
  const data = new TextEncoder().encode(canonicalGroupJson(value));
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", data))].map(n=>n.toString(16).padStart(2,"0")).join("");
}
export function parseHouseReviewedCatalog(value:unknown):HouseReviewedCatalog|null {
  try {
    if (!record(value) || canonicalGroupJson(value) !== canonicalGroupJson(bindings)) return null;
    const catalog = value as unknown as HouseReviewedCatalog;
    if (catalog.schemaVersion !== 2 || catalog.runs.length !== 32 ||
      catalog.sourcePolicySha256 !== HOUSE_GROUP_POLICY_SHA || catalog.sourceCatalogSha256 !== HOUSE_GROUP_SOURCE_CATALOG_SHA ||
      catalog.qualificationSha256 !== HOUSE_GROUP_QUALIFICATION_SHA) return null;
    const oldRuns = catalog.runs.filter(r=>!r.polling).map(({polling,conditionalShareRunId,conditionalShareSha256,
      conditionalShareContentSha256,sourcePolicySha256,sourceCatalogSha256,qualificationSha256,matchedBaselineRunId,...old})=>old);
    if (!parseHouseGroupCatalog({...catalog,schemaVersion:1,runs:oldRuns})) return null;
    if (catalog.runs.filter(r=>r.polling).some(r=>r.sourcePolicySha256 !== HOUSE_POLLING_POLICY_SHA ||
      r.sourceCatalogSha256 !== HOUSE_POLLING_CATALOG_SHA || r.qualificationSha256 !== HOUSE_POLLING_AUDIT_SHA)) return null;
    return catalog;
  } catch { return null; }
}
export async function parseHouseReviewedPublication(value:unknown):Promise<HouseReviewedPublication|null> {
  try {
    if (!record(value) || value.schemaVersion !== 1 || value.status !== "available" || value.family !== "house_groups" ||
      !record(value.entry)) return null;
    const catalog = parseHouseReviewedCatalog(value.catalog);
    const suppliedEntry=value.entry;
    const entry = catalog?.runs.find(r=>r.id === suppliedEntry.id);
    if (!entry || canonicalGroupJson(entry) !== canonicalGroupJson(value.entry)) return null;
    const result = entry.polling ? parseHousePollingGroupView(value.result) : parseHouseGroupView(value.result);
    if (!result || result.id !== entry.id || result.cycle !== entry.cycle || result.legacyRunId !== entry.legacyRunId ||
      result.provenance.sourcePolicySha256 !== entry.sourcePolicySha256 ||
      result.configuration !== (entry.polling ? `${entry.fundamentalsId}_polling_v1` : `${entry.fundamentalsId}-loading-1`) ||
      await reviewedDigest(result) !== entry.viewSha256) return null;
    if (entry.polling && (!("polling" in result) || result.conditionalShareRunId !== entry.conditionalShareRunId ||
      result.conditionalShareSha256 !== entry.conditionalShareSha256)) return null;
    const geography = value.geography as DistrictGeography;
    if (geography?.metadata?.sourceSha256 !== entry.geographySha256 || geography.metadata.mapVintage !== entry.mapVintage ||
      geography.geojson.features.length !== 435 || await reviewedDigest(geography.geojson) !== entry.geographyContentSha256) return null;
    const joined = joinDistrictGeography(result.races.map(race=>({race:groupRaceForMap(race),districtKey:race.districtKey})),
      {cycle:result.cycle,mapVintage:entry.mapVintage},geography);
    if (joined.status !== "ready" || joined.matches.length !== 435) return null;
    return {schemaVersion:1,status:"available",family:"house_groups",catalog:catalog!,entry,result,geography};
  } catch { return null; }
}

export function houseReviewedCatalog():HouseReviewedCatalog {
  const value=parseHouseReviewedCatalog(structuredClone(bindings));
  if(!value)throw Error("Reviewed bindings are unavailable.");
  return value;
}
