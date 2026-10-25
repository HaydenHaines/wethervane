const pair = (id, first, second) => ({ id, raceId:`race-${id}`, eventId:`event-${id}`, contestId:`contest-${id}`,
  electionDate:"2032-11-02", first:{id:`person-${id}-a`,name:first}, second:{id:`person-${id}-b`,name:second} });
const report = (id, profile="Invented 2032 counted-ballot report") => ({ id, eventId:`event-${id}`, date:"2032-11-02",
  name:`${id} administrative report`, releaseId:`invented-release-${id}`, profile,
  definition:"Counted ballots in this invented administrative release", sourceQualityFlags:id.endsWith("2")?["revision unknown"]:[],
  sourceVariance:null, effectProvenance:{release:"fresh illustrative release effect",state:"same-row illustrative state effect"} });
const interval = (level, lower, upper, covered) => ({level,lower,upper,width:upper-lower,covered});
export const scoredPair = (observed=.53) => ({state:"scored",undefinedMass:0,observedShare:observed,crps:.08,wis:.12,
  intervals:[interval(.8,.4,.65,true)], leading:{probability:.52,tieProbability:.03,observed:true,brier:.2304,logLoss:.6539,logLossInfinite:false},
  shape:{near50:true,probMarginWithin2pp:.12,probMarginAtLeast10pp:.61,expectedSignedMarginPp:3,medianSignedMarginPp:2}});
export const missingPair = () => ({state:"outcome_missing",undefinedMass:0,observedShare:null,crps:null,wis:null,
  intervals:[interval(.8,.3,.7,null)], leading:{probability:.48,tieProbability:.02,observed:null,brier:null,logLoss:null,logLossInfinite:false},reason:"No recorded pair share."});
export const scoredReport = (observed=0) => ({state:"scored",eventPresent:true,undefinedMass:0,observedCount:observed,
  predictedMean:4,crps:3,wis:5,intervals:[interval(.8,0,10,true)],logLoss:2.1,logLossInfinite:false});
export const inactiveReport = () => ({state:"inactive_event",eventPresent:false,undefinedMass:null,observedCount:null,
  predictedMean:null,crps:null,wis:null,intervals:[],logLoss:null,logLossInfinite:false,reason:"Runoff did not occur."});
const metricUnit = id => id.startsWith("report_") ? id === "report_log" ? "dimensionless" : "report_count"
  : id.startsWith("leading_") ? "dimensionless" : "share_fraction";
const metricValue = id => id.startsWith("report_") ? 3 : id === "vector_energy" ? .09 : .08;
const ids = ["pair_crps","pair_wis","leading_brier","leading_log","vector_energy","report_crps","report_wis","report_log"];
function metrics(cycle, modelResult) {
  const out={};
  for(const id of ids){
    const expected=id.startsWith("report_")?cycle.reportRoster.map(r=>r.id):id==="vector_energy"?["whole_vector"]:cycle.pairRoster.map(p=>p.id);
    const scored=id.startsWith("report_")?expected.filter(key=>modelResult.reports[key]?.state==="scored")
      :id==="vector_energy"?modelResult.energy.state==="scored"?expected:[]
      :expected.filter(key=>modelResult.pairs[key]?.state==="scored");
    const state=scored.length===0?"unavailable":scored.length===expected.length?"complete":"partial";
    out[id]={state,value:state==="unavailable"?null:metricValue(id),infinite:false,scoredIds:scored,
      binding:{target:id.startsWith("report_")?"named administrative count":id==="vector_energy"?"whole ordered named-pair share vector":"named-pair share or strict lead",
        orderedRosterIds:expected,cutoff:cycle.cutoff,outcomeView:cycle.outcomeView,metricDefinition:`invented-${id}-v1`,weighting:"fixed equal expected roster",
        unit:metricUnit(id),...(id.startsWith("report_")?{reportProfile:cycle.reportRoster[0]?.profile}: {})},
      reason:state==="unavailable"?"Required results missing.":undefined,
      precision:{status:"not_reported",note:"No simulation precision certificate was supplied."}};
  }
  return out;
}
function cycle(id,pairCount,reportCount){
  const c={id,label:`${id} invented cycle`,cutoff:`${id}-10-19T00:00:00Z`,outcomeView:`invented-${id}-outcomes`,
    pairRoster:Array.from({length:pairCount},(_,i)=>pair(`${id}-pair-${i+1}`,`Alex ${id}-${i+1}`,`Blair ${id}-${i+1}`)),
    reportRoster:Array.from({length:reportCount},(_,i)=>report(`${id}-report-${i+1}`)),models:{}};
  for(const modelId of ["count","poll"]){
    const r={pairs:{},reports:{},energy:{state:"withheld",value:null,undefinedMass:0,missingObservedIds:[],positiveWeightMissingIds:[],
      orderedPairIds:c.pairRoster.map(p=>p.id),rows:8,weightSum:1,reason:"A required pair coordinate has no recorded outcome."},metrics:{}};
    c.pairRoster.forEach((p,i)=>r.pairs[p.id]=i===0?scoredPair():missingPair());
    c.reportRoster.forEach((q,i)=>r.reports[q.id]=i===0?scoredReport():inactiveReport());
    r.energy.missingObservedIds=c.pairRoster.slice(1).map(p=>p.id);
    r.metrics=metrics(c,r);c.models[modelId]=r;
  }
  return c;
}
export function evaluationFixture(){return {schema:"PrivateEvaluationScoreboardDraftV1",
  provenance:{kind:"invented_fixture",label:"Invented evaluation examples for frontend testing",declaredSourceHashes:["invented-declaration-abc"]},
  label:"Historical forecast quality · illustrative",status:{run:"complete",evidence:"experimental",adoption:"not_qualified",note:"Interface evidence only; no real posterior or empirical forecast score."},
  models:[{id:"count",name:"Illustrative count stack",ingredients:["Population and participation ingredients","No race-poll input"],informationKey:"counts"},
    {id:"poll",name:"Illustrative poll stack",ingredients:["Population and participation ingredients","Race-poll input"],informationKey:"polls"}],
  cycles:[cycle("2032",2,2),cycle("2036",1,1)]};}
export function completeFixture(){const d=evaluationFixture();const c=d.cycles[0];
  for(const r of Object.values(c.models)){
    c.pairRoster.forEach(p=>r.pairs[p.id]=scoredPair());c.reportRoster.forEach(q=>r.reports[q.id]=scoredReport());
    r.energy={state:"scored",value:.09,undefinedMass:0,missingObservedIds:[],positiveWeightMissingIds:[],orderedPairIds:c.pairRoster.map(p=>p.id),rows:8,weightSum:1};
    r.metrics=metrics(c,r);
  }return d;}
export function refreshMetrics(d,cycleIndex=0,modelId="count") {const c=d.cycles[cycleIndex];c.models[modelId].metrics=metrics(c,c.models[modelId]);return d;}
