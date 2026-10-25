#!/usr/bin/env -S node --experimental-strip-types
/** Import the sealed FP packet through a new adapter, preserving all predecessor bytes. */
import { createHash } from "node:crypto";
import { readFile, realpath, mkdir, mkdtemp, writeFile, rename, rm } from "node:fs/promises";
import { dirname, join, resolve, relative, isAbsolute } from "node:path";
import { pathToFileURL } from "node:url";
import { parseHousePollingGroupResult, HOUSE_POLLING_POLICY_SHA, HOUSE_POLLING_CATALOG_SHA,
  HOUSE_POLLING_AUDIT_SHA, HOUSE_POLLING_INFORMATION_MODE } from "../lib/house-polling-result.ts";
import { parseHousePollingShares } from "../lib/house-polling-shares.ts";
import { canonicalGroupJson, HOUSE_GROUP_POLICY_SHA, HOUSE_GROUP_QUALIFICATION_SHA, HOUSE_GROUP_SOURCE_CATALOG_SHA } from "../lib/house-group-catalog.ts";
import { loadHouseGroupPublication } from "../lib/house-group-publication.ts";

const BASE = "research/house-polling-benchmark-20261024/frontend-results-v1";
const hash = bytes=>createHash("sha256").update(bytes).digest("hex");
const encoded = value=>Buffer.from(canonicalGroupJson(value));
const require = (value,message)=>{if(!value)throw Error(message);};
const same = (a,b)=>canonicalGroupJson(a)===canonicalGroupJson(b);
function csv(text) {
  const rows = [];let row=[],field="",quoted=false;
  for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;}
    else if(c===','&&!quoted){row.push(field);field="";}else if(c==='\n'&&!quoted){row.push(field.replace(/\r$/,""));rows.push(row);row=[];field="";}else field+=c;}
  if(field||row.length){row.push(field);rows.push(row);}require(!quoted,"Unclosed CSV quote.");
  const headers=rows.shift();return rows.filter(r=>r.some(Boolean)).map(r=>{require(r.length===headers.length,"Invalid CSV row.");return Object.fromEntries(headers.map((h,i)=>[h,r[i]]));});
}
function support(result, aliases, polling) {
  return result.races.map(r=>({id:r.id,districtKey:r.districtKey,ruleFamily:r.ruleFamily,structural:r.structural,
    cutoff:r.nominalCutoff,key:r.comparisonKey,mode:r.supportMode,observed:r.observed,score:r.groupScoreEligible,core:r.voteShareReferenceAvailable,
    groups:r.groups.map(g=>({id:g.id,label:g.label,incumbent:g.incumbentGroup,singleton:g.namedSingletonResolutionEligible,
      members:g.members.map(m=>{const alias=aliases.find(a=>a.race_id===r.id&&a.party===g.id&&a.publisher_name===m.name);
        if(polling&&m.sourceDisplayName!==undefined)require(alias&&alias.roster_name.trim()===m.sourceDisplayName.trim(),"Unreviewed display alias.");
        return {id:m.id,kind:m.kind,name:polling&&alias?alias.roster_name.trim():m.name};})}))}));
}
function assertScoreRows(metrics, table, cycle, implementation) {
  for(const score of metrics){
    const rows=table.filter(r=>Number(r.cycle)===cycle&&r.implementation===implementation&&r.variant==="primary"&&r.subset===score.subset);
    require(rows.length===1,"Missing or ambiguous score-table row.");
    for(const [key,value] of Object.entries(score)){
      if(typeof value==="number")require(Number.isFinite(value)&&Math.abs(Number(rows[0][key])-value)<=1e-11,"Score table differs: "+key);
      else if(typeof value==="boolean")require(rows[0][key]===String(value).replace(/^./,s=>s.toUpperCase()),"Score qualification differs.");
      else require(rows[0][key]===value,"Score identity differs: "+key);
    }
  }
}
export async function importHousePolling({sourceRoot,groupDirectory,legacyDirectory,outputDirectory}) {
  const root=await realpath(sourceRoot),groups=await realpath(groupDirectory),legacy=await realpath(legacyDirectory),output=resolve(outputDirectory);
  require(relative(root,output).startsWith("../"),"Output must be outside scientific checkout.");
  const verified=new Map();
  async function pinned(reference){
    require(reference&&typeof reference.path==="string"&&!isAbsolute(reference.path)&&!reference.path.split(/[\\/]/).includes("..")&&/^[a-f0-9]{64}$/.test(reference.sha256),"Invalid source pin.");
    const path=await realpath(join(root,reference.path));require(path.startsWith(root+"/"),"Source pin leaves checkout.");
    require(!verified.has(path)||verified.get(path)===reference.sha256,"Conflicting source pin.");
    const bytes=await readFile(path);require(hash(bytes)===reference.sha256,"Source fingerprint changed: "+reference.path);
    verified.set(path,reference.sha256);return bytes;
  }
  async function nested(value){if(Array.isArray(value)){for(const child of value)await nested(child);}else if(value&&typeof value==="object"){
    if(typeof value.path==="string"&&typeof value.sha256==="string")await pinned(value);
    for(const child of Object.values(value))await nested(child);}}
  async function document(reference){const value=JSON.parse(await pinned(reference));await nested(value);return value;}
  const source=await document({path:BASE+"/catalog.json",sha256:HOUSE_POLLING_CATALOG_SHA});
  require(source.schema==="HouseHistoricalPollingReviewedCatalogV1"&&source.status==="reviewed_experimental_research"&&
    source.entries.length===16&&source.matched_pairs.length===16&&source.current2026===false&&source.public_chamber_control_qualified===false,
    "Polling catalog scope changed.");
  require(source.source_policy.sha256===HOUSE_POLLING_POLICY_SHA&&source.output_review.sha256===HOUSE_POLLING_AUDIT_SHA,"Polling authority changed.");
  const policy=await document(source.source_policy),review=await document(source.output_review),admission=await document(source.source_admission);
  require(policy.schema==="HouseHistoricalPollingSourceCapabilityPolicyV1"&&policy.information_mode===HOUSE_POLLING_INFORMATION_MODE&&
    policy.capabilities.polling===true&&policy.capabilities.public_chamber_control===false&&policy.capabilities.tipping===false&&
    policy.capabilities.current2026===false&&policy.capabilities.individual_O_winners===false&&policy.capabilities.full435_vote_shares===false,
    "Polling policy capabilities changed.");
  require(policy.inherited_source_policies.length===2&&policy.inherited_source_policies[0].sha256===HOUSE_GROUP_POLICY_SHA,
    "Predecessor policy inheritance changed.");
  for(const inherited of policy.inherited_source_policies){const old=await document(inherited);require(old.capabilities.polling===false,"Predecessor polling capability changed.");}
  require(review.schema==="HousePollingOutputAuditV1"&&review.status==="passed"&&Object.values(review.checks).every(v=>v===true),"Output audit did not pass.");
  await document(source.protocol);
  await document(review.results);
  await document(admission.normalized);
  const exception=await document(admission.identity_exceptions);
  require(exception.schema==="HousePollIdentityExceptionsV3","Unreviewed display alias receipt.");
  const manifest=JSON.parse(await readFile(join(groups,"publication.json"))),oldBytes=await readFile(join(groups,manifest.catalog.file));
  require(hash(oldBytes)===manifest.catalog.sha256,"Predecessor collection changed.");
  const oldCatalog=JSON.parse(oldBytes),files=new Map(),artifacts=[],runs=[],matchedPairs=[];
  const legacyManifest=JSON.parse(await readFile(join(legacy,"publication.json"))),legacyBytes=await readFile(join(legacy,legacyManifest.catalog.file));
  require(hash(legacyBytes)===legacyManifest.catalog.sha256,"Legacy share collection changed.");
  const legacyCatalog=JSON.parse(legacyBytes);
  for(const old of oldCatalog.runs){
    await loadHouseGroupPublication(groups,legacy,old.id);
    const share=legacyCatalog.runs.find(r=>r.id===old.legacyRunId);
    const shareBytes=await readFile(join(legacy,share.result.file));require(hash(shareBytes)===old.legacyResultSha256,"Legacy share bytes differ.");
    const {view,geography,...identity}=old;
    runs.push({...identity,polling:false,conditionalShareRunId:old.legacyRunId,conditionalShareSha256:old.legacyResultSha256,
      conditionalShareContentSha256:hash(encoded(JSON.parse(shareBytes))),sourcePolicySha256:HOUSE_GROUP_POLICY_SHA,
      sourceCatalogSha256:HOUSE_GROUP_SOURCE_CATALOG_SHA,qualificationSha256:HOUSE_GROUP_QUALIFICATION_SHA,matchedBaselineRunId:null});
    artifacts.push({id:old.id,view,geography});
  }
  const groupScores=csv((await pinned(source.tables.find(r=>r.path.endsWith("/group-cycle-scores.csv")))).toString());
  const shareScores=csv((await pinned(source.tables.find(r=>r.path.endsWith("/share-cycle-scores.csv")))).toString());
  for(const entry of source.entries){
    const raw=await document(entry.full_roster),shareBytes=await pinned(entry.conditional_shares),shareRaw=JSON.parse(shareBytes);
    await nested(shareRaw);
    const result=parseHousePollingGroupResult(raw,{sourcePolicySha256:HOUSE_POLLING_POLICY_SHA}),shares=parseHousePollingShares(shareRaw);
    require(result&&shares&&result.id===entry.id&&result.cycle===entry.cycle&&result.configuration===entry.configuration&&
      shares.run.id===result.conditionalShareRunId&&result.conditionalShareSha256===entry.conditional_shares.sha256&&
      same(result.pollingCoverage,shares.pollingCoverage),"Invalid reviewed polling result or shares.");
    const oldId=entry.id.replace("-full435-polls-v1","-full435-v1"),old=oldCatalog.runs.find(r=>r.id===oldId);
    require(old&&old.resultSha256===entry.baseline_full_roster.sha256&&old.legacyResultSha256===entry.baseline_conditional_shares.sha256,
      "Baseline scientific identity changed.");
    const baseline=await loadHouseGroupPublication(groups,legacy,oldId);
    require(same(support(baseline.result,exception.aliases,false),support(result,exception.aliases,true)),"F/FP roster, truth or score masks differ.");
    require(same(shares.races.map(r=>r.id).sort(),result.races.filter(r=>r.voteShareReferenceAvailable).map(r=>r.id).sort()),"Conditional share support differs.");
    const pair=source.matched_pairs.filter(p=>p.baseline.sha256===old.resultSha256&&p.combined.sha256===entry.full_roster.sha256);
    require(pair.length===1&&pair[0].cycle===entry.cycle&&pair[0].group_rows===435&&pair[0].share_rows===shares.races.length&&
      result.races.every(r=>r.comparisonKey===pair[0].comparison_key),"Reviewed matched pair differs.");
    const implementation=old.fundamentalsId.replace(/^house_/,"senate_");
    const auditCell=review.group_cells.filter(c=>c.cycle===entry.cycle&&c.implementation===implementation&&c.variant==="primary");
    require(auditCell.length===1&&same(auditCell[0].diagnostics,raw.metrics.diagnostics),"Group diagnostics differ from actual-output audit.");
    const shareCell=review.share_cells.filter(c=>c.cycle===entry.cycle&&c.implementation===implementation&&c.variant==="primary");
    require(shareCell.length===1,"Share audit cell missing.");
    const predictionReceipt=await document(shareCell[0].prediction_receipt);
    require(predictionReceipt.plan.sha256===shares.run.configurationSha256&&
      predictionReceipt.outputs.some(p=>p.sha256===shareRaw.jointDraws.sourceSha256),"Conditional share prediction bank differs.");
    assertScoreRows(raw.metrics.race_scores,groupScores,entry.cycle,implementation);
    assertScoreRows(raw.metrics.share_scores,shareScores,entry.cycle,implementation);
    const viewBytes=encoded(result),viewSha256=hash(viewBytes),view={file:`view-${viewSha256}.json`,sha256:viewSha256};files.set(view.file,viewBytes);
    const sharesRef={file:`shares-${entry.conditional_shares.sha256}.json`,sha256:entry.conditional_shares.sha256};files.set(sharesRef.file,shareBytes);
    runs.push({id:result.id,cycle:result.cycle,fundamentalsId:old.fundamentalsId,label:old.label+" + historical polls",
      resultSha256:entry.full_roster.sha256,viewSha256,legacyRunId:old.legacyRunId,legacyResultSha256:old.legacyResultSha256,
      mapVintage:old.mapVintage,geographySha256:old.geographySha256,geographyContentSha256:old.geographyContentSha256,
      polling:true,conditionalShareRunId:shares.run.id,conditionalShareSha256:entry.conditional_shares.sha256,
      conditionalShareContentSha256:hash(encoded(shares)),sourcePolicySha256:HOUSE_POLLING_POLICY_SHA,
      sourceCatalogSha256:HOUSE_POLLING_CATALOG_SHA,qualificationSha256:HOUSE_POLLING_AUDIT_SHA,matchedBaselineRunId:old.id});
    artifacts.push({id:result.id,view,shares:sharesRef,geography:old.geography});
    matchedPairs.push({baselineRunId:old.id,pollingRunId:result.id,comparisonKey:pair[0].comparison_key,groupRows:435,shareRows:shares.races.length});
  }
  require(runs.length===32&&new Set(runs.map(r=>r.id)).size===32,"Reviewed collection identities differ.");
  const bindings={schemaVersion:2,qualificationSha256:HOUSE_GROUP_QUALIFICATION_SHA,sourcePolicySha256:HOUSE_GROUP_POLICY_SHA,
    sourceCatalogSha256:HOUSE_GROUP_SOURCE_CATALOG_SHA,initialRunId:oldCatalog.initialRunId,runs,matchedPairs,
    aliases:exception.aliases.map(a=>({raceId:a.race_id,party:a.party,baselineName:a.roster_name.trim(),pollingName:a.publisher_name,receiptSha256:admission.identity_exceptions.sha256}))};
  const catalogBytes=encoded({bindings,artifacts}),catalogSha256=hash(catalogBytes),catalogRef={file:`catalog-${catalogSha256}.json`,sha256:catalogSha256};
  files.set(catalogRef.file,catalogBytes);files.set("reviewed-bindings.json",encoded(bindings));
  files.set("import-receipt.json",encoded({sourceCatalogSha256:HOUSE_POLLING_CATALOG_SHA,sourcePolicySha256:HOUSE_POLLING_POLICY_SHA,
    outputAuditSha256:HOUSE_POLLING_AUDIT_SHA,verifiedSourcePins:[...verified].map(([path,sha256])=>({path:relative(root,path),sha256})),
    matchedPairs:matchedPairs.length,oldArtifactsChanged:false,publicChamberControl:false,current2026:false}));
  const parent=await realpath(dirname(output));
  require(parent===dirname(output),"Output parent must not traverse symlinks.");
  require(relative(root,join(parent,output.split("/").pop())).startsWith("../"),"Resolved output must be outside scientific checkout.");
  const stage=await mkdtemp(join(dirname(output),".house-polling-"));
  try {
    for(const [name,bytes] of files)await writeFile(join(stage,name),bytes,{flag:"wx"});
    await writeFile(join(stage,"publication.json"),encoded({schemaVersion:2,status:"available",catalog:catalogRef}),{flag:"wx"});
    await mkdir(output,{recursive:true});require(await realpath(output)===output,"Output must not be symlink.");
    for(const [name,bytes] of files){try{await writeFile(join(output,name),bytes,{flag:"wx"});}
      catch(error){if(error.code!=="EEXIST")throw error;require(hash(await readFile(join(output,name)))===hash(bytes),"Conflicting polling artifact.");}}
    await rename(join(stage,"publication.json"),join(output,"publication.json"));
    return {runs:runs.length,pollingRuns:16,matchedPairs:16,verifiedSourcePins:verified.size,catalogSha256};
  } finally {await rm(stage,{recursive:true,force:true});}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
  const args=process.argv.slice(2);require(args.length===8&&args[0]==="--source-root"&&args[2]==="--group-dir"&&args[4]==="--legacy-dir"&&args[6]==="--output-dir",
    "Expected --source-root --group-dir --legacy-dir --output-dir.");
  console.log(JSON.stringify(await importHousePolling({sourceRoot:args[1],groupDirectory:args[3],legacyDirectory:args[5],outputDirectory:args[7]})));
}
