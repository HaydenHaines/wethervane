import { canonicalGroupJson, parseHouseReviewedCatalog, parseHouseReviewedPublication, type HouseReviewedCatalog as HouseGroupCatalog, type HouseReviewedPublication as HouseGroupPublication } from "./house-reviewed-catalog.ts";
import { verifyHouseGroupComparison } from "./house-reviewed-comparison.ts";
export const HOUSE_GROUP_SAVED_VIEWS_STORAGE_KEY = "ballotmancy.house.groups.saved-views.v1";
export const MAX_HOUSE_GROUP_SAVED_VIEWS = 30;
export interface HouseGroupSavedBinding {
  runId:string; runLabel:string; cycle:number; configuration:string; targetId:string; comparisonFamily:string;
  resultSha256:string; viewSha256:string; sourcePanelSha256:string; predictionSha256:string; auditSha256:string;
  sourcePolicySha256:string; sourceCatalogSha256:string; qualificationSha256:string;
  mapVintage:string; geographySha256:string; geographyContentSha256:string; legacyRunId:string; legacyResultSha256:string;
}
export interface HouseGroupSavedView extends HouseGroupSavedBinding {
  id:string; label:string; savedAt:string; raceId:string; mapView:"forecast"|"actual"; comparison?:HouseGroupSavedBinding;
}
export type HouseGroupSavedVerification = {status:"verified";reason?:never}|{status:"unavailable";reason:string};
const record=(v:unknown):v is Record<string,unknown>=>!!v&&typeof v==="object"&&!Array.isArray(v);
const text=(v:unknown,max=300):v is string=>typeof v==="string"&&v.trim()===v&&v.length>0&&v.length<=max&&!/[\u0000-\u001f\u007f]/.test(v);
const hash=(v:unknown):v is string=>typeof v==="string"&&/^[a-f0-9]{64}$/.test(v);
const hashFields=["resultSha256","viewSha256","sourcePanelSha256","predictionSha256","auditSha256","sourcePolicySha256","sourceCatalogSha256","qualificationSha256","geographySha256","geographyContentSha256","legacyResultSha256"] as const;
function binding(value:unknown):HouseGroupSavedBinding|null {
  if(!record(value)||!["runId","runLabel","configuration","targetId","comparisonFamily","mapVintage","legacyRunId"].every(k=>text(value[k]))||!Number.isInteger(value.cycle)||(value.cycle as number)<1900||(value.cycle as number)>3000||!hashFields.every(k=>hash(value[k])))return null;
  return {runId:value.runId as string,runLabel:value.runLabel as string,cycle:value.cycle as number,configuration:value.configuration as string,targetId:value.targetId as string,comparisonFamily:value.comparisonFamily as string,mapVintage:value.mapVintage as string,legacyRunId:value.legacyRunId as string,...Object.fromEntries(hashFields.map(k=>[k,value[k]]))} as HouseGroupSavedBinding;
}
function savedView(value:unknown):HouseGroupSavedView|null {
  const bound=binding(value);
  if(!bound||!record(value)||!text(value.id,160)||!text(value.label,80)||!text(value.savedAt)||!Number.isFinite(Date.parse(value.savedAt))||new Date(value.savedAt).toISOString()!==value.savedAt||!text(value.raceId)||!new RegExp(`^${bound.cycle}:[A-Z]{2}-[0-9]{2}:house:regular$`).test(value.raceId)||!["forecast","actual"].includes(value.mapView as string))return null;
  const comparison=value.comparison===undefined?undefined:binding(value.comparison);
  if(comparison===null||(comparison&&(comparison.runId===bound.runId||comparison.cycle!==bound.cycle||comparison.mapVintage!==bound.mapVintage||comparison.comparisonFamily!==bound.comparisonFamily)))return null;
  return {...bound,id:value.id,label:value.label,savedAt:value.savedAt,raceId:value.raceId,mapView:value.mapView as HouseGroupSavedView["mapView"],...(comparison?{comparison}:{})};
}
/** Valid stale entries remain readable and deletable; they are never silently rebound. */
export function parseHouseGroupSavedViews(raw:string|null):HouseGroupSavedView[]{
  try{if(typeof raw!=="string"||raw.length>256000)return[];const value=JSON.parse(raw);if(!record(value)||value.schemaVersion!==1||!Array.isArray(value.views))return[];const seen=new Set<string>();return value.views.slice(0,MAX_HOUSE_GROUP_SAVED_VIEWS).flatMap(v=>{const parsed=savedView(v);if(!parsed||seen.has(parsed.id))return[];seen.add(parsed.id);return[parsed];});}catch{return[];}
}
export function serializeHouseGroupSavedViews(views:HouseGroupSavedView[]):string{
  if(views.length>MAX_HOUSE_GROUP_SAVED_VIEWS)throw Error("Too many saved group views.");const serialized=JSON.stringify({schemaVersion:1,views});const parsed=parseHouseGroupSavedViews(serialized);if(parsed.length!==views.length)throw Error("Invalid saved group views.");return JSON.stringify({schemaVersion:1,views:parsed});
}
function publicationBinding(p:HouseGroupPublication):HouseGroupSavedBinding {
  const r=p.result,e=p.entry,c=p.catalog;
  return {runId:r.id,runLabel:e.label,cycle:r.cycle,configuration:r.configuration,targetId:r.targetId,comparisonFamily:r.comparisonFamily,resultSha256:e.resultSha256,viewSha256:e.viewSha256,sourcePanelSha256:r.provenance.sourcePanelSha256,predictionSha256:r.provenance.predictionSha256,auditSha256:r.provenance.auditSha256,sourcePolicySha256:e.sourcePolicySha256,sourceCatalogSha256:e.sourceCatalogSha256,qualificationSha256:e.qualificationSha256,mapVintage:e.mapVintage,geographySha256:e.geographySha256,geographyContentSha256:e.geographyContentSha256,legacyRunId:e.legacyRunId,legacyResultSha256:e.legacyResultSha256};
}
function sameBinding(a:HouseGroupSavedBinding,b:HouseGroupSavedBinding):boolean {const {runLabel:ignoredA,...left}=a,{runLabel:ignoredB,...right}=b;return canonicalGroupJson(left)===canonicalGroupJson(right);}
export function houseGroupSavedViewAvailability(view:HouseGroupSavedView,catalog:HouseGroupCatalog):{status:"catalog_compatible"|"unavailable";reason?:string}{
  const parsed=savedView(view),c=parseHouseReviewedCatalog(catalog);
  if(!parsed||!c)return{status:"unavailable",reason:"The saved view or current collection could not be verified."};
  for(const bound of [parsed,...(parsed.comparison?[parsed.comparison]:[])]){
    const entry=c.runs.find(e=>e.id===bound.runId);
    if(!entry||bound.targetId!==(entry.polling?"house_aggregate_polling_group_winner_completion_v1":"house_aggregate_group_winner_completion_v1")||bound.comparisonFamily!=="house_ultimate_nominee_group_winner_v1"||bound.resultSha256!==entry.resultSha256||bound.viewSha256!==entry.viewSha256||bound.cycle!==entry.cycle||bound.configuration!==(entry.polling?`${entry.fundamentalsId}_polling_v1`:`${entry.fundamentalsId}-loading-1`)||bound.mapVintage!==entry.mapVintage||bound.geographySha256!==entry.geographySha256||bound.geographyContentSha256!==entry.geographyContentSha256||bound.legacyRunId!==entry.legacyRunId||bound.legacyResultSha256!==entry.legacyResultSha256||bound.qualificationSha256!==entry.qualificationSha256||bound.sourcePolicySha256!==entry.sourcePolicySha256||bound.sourceCatalogSha256!==entry.sourceCatalogSha256)return{status:"unavailable",reason:"An exact saved result, source qualification or map binding is no longer available."};
  }
  return{status:"catalog_compatible"};
}
export async function verifyHouseGroupSavedView(view:HouseGroupSavedView,primary:HouseGroupPublication,comparison?:HouseGroupPublication|null):Promise<HouseGroupSavedVerification>{
  const unavailable=(reason:string):HouseGroupSavedVerification=>({status:"unavailable",reason});
  try{
    const parsed=savedView(view),p=await parseHouseReviewedPublication(primary);
    if(!parsed||!p||houseGroupSavedViewAvailability(parsed,p.catalog).status!=="catalog_compatible"||!sameBinding(binding(parsed)!,publicationBinding(p))||!p.result.races.some(r=>r.id===parsed.raceId))return unavailable("The exact saved group result, district or immutable evidence bindings could not be verified.");
    if(parsed.comparison){if(!comparison)return unavailable("The exact saved comparison result must be loaded before restoring this view.");const other=await parseHouseReviewedPublication(comparison);if(!other||!sameBinding(parsed.comparison,publicationBinding(other))||(await verifyHouseGroupComparison(p,other)).status!=="comparable")return unavailable("The exact saved comparison or its shared outcome target could not be verified.");}
    return{status:"verified"};
  }catch{return unavailable("The saved group view could not be verified.");}
}
export async function createHouseGroupSavedView(label:string,publication:HouseGroupPublication,selection:{raceId:string;mapView:"forecast"|"actual";comparison?:HouseGroupPublication|null}):Promise<HouseGroupSavedView>{
  if(!text(label.trim(),80))throw Error("Name the saved group view using 1 to 80 characters.");
  const p=await parseHouseReviewedPublication(publication);if(!p)throw Error("The displayed publication could not be verified.");
  const view:HouseGroupSavedView={...publicationBinding(p),id:crypto.randomUUID(),label:label.trim(),savedAt:new Date().toISOString(),raceId:selection.raceId,mapView:selection.mapView,...(selection.comparison?{comparison:publicationBinding(selection.comparison)}:{})};
  if((await verifyHouseGroupSavedView(view,p,selection.comparison)).status!=="verified")throw Error("The exact selected group view could not be verified.");return view;
}
