#!/usr/bin/env python3
"""Combine only independently reviewed 2018/2022 report JSONs; opens no raw banks or labels."""
import hashlib, json, math
from pathlib import Path
HERE=Path(__file__).resolve().parent
INPUTS={
 "2018":{"path":HERE/"independent-score-review-2018-r2.json","sha256":"c320ee29350de4e077a2d39bb9c16862757c330d37c53629444a8f313350eaef"},
 "2022":{"path":HERE/"independent-score-review-2022.json","sha256":"8e2f3ee714cbfda0b94b0f6038b6e582e6c4627ccf41e5ca695d9bef564a9e43"},
}
FIELDS=("share_crps","share_wis","win_brier","win_log_score","margin_mae_pp","coverage80","width80","coverage95","width95")
METRICS=("share_crps","win_brier","win_log_score")
reports={}
for year,item in INPUTS.items():
 data=item["path"].read_bytes()
 assert hashlib.sha256(data).hexdigest()==item["sha256"]
 reports[int(year)]=json.loads(data)
cycles=[reports[y]["per_cycle"][0] for y in (2018,2022)]
assert all(c["target"] in (2018,2022) and c["tier"]=="A" for c in cycles)
arms=("H0","F0","G01")
means={}
for arm in arms:
 means[arm]={}
 for metric in FIELDS:
  values=[c["arms"][arm]["strata"]["all"][metric] for c in cycles]
  means[arm][metric]=sum(values)/2
delta={k:means["G01"][k]-means["F0"][k] for k in FIELDS}
primary=[]
for stratum in ("all","competitive","near_even"):
 for metric in METRICS:
  rows=[]
  for c in cycles:
   f0,g01=(c["arms"][a]["strata"][stratum] for a in ("F0","G01"))
   if not f0["n"] or not g01["n"]:
    rows=[];break
   bytarget={x["metric"]:x for x in reports[c["target"]]["primary_G01_minus_F0"]["per_cycle"]
     if x["stratum"]==stratum}
   row=bytarget[metric]
   rows.append(row)
  if not rows:
   primary.append({"tier":"A","stratum":stratum,"metric":metric,"status":"unavailable_empty_mask",
     "G01_minus_F0":None,"combined_mcse":None,"loss":None,"loss_beyond_two_mcse":None,"precision_resolved":False})
   continue
  d=sum(r["G01_minus_F0"] for r in rows)/2
  e=math.sqrt(sum(r["combined_mcse"]**2 for r in rows))/2
  resolved=all(r["precision_resolved"] for r in rows)
  primary.append({"tier":"A","stratum":stratum,"metric":metric,"status":"available",
    "cycle_deltas":[{"target":r["target"],"delta":r["G01_minus_F0"],"combined_mcse":r["combined_mcse"]} for r in rows],
    "G01_minus_F0":d,"combined_mcse":e,"loss":d>0,
    "loss_beyond_two_mcse":d>2*e if resolved else None,"precision_resolved":resolved})
# Practical CRPS guard is a descriptive aid only; compare A mean delta with F0 A mean.
practical={}
for stratum in ("all","competitive"):
 vals=[c["arms"]["F0"]["strata"][stratum]["share_crps"] for c in cycles]
 d=primary[[x["stratum"]=="all" and x["metric"]=="share_crps" or x["stratum"]=="competitive" and x["metric"]=="share_crps" for x in primary].index(True)] if False else None
 row=next(x for x in primary if x["stratum"]==stratum and x["metric"]=="share_crps")
 threshold=max(.0005,.02*(sum(vals)/2))
 practical[stratum]={"absolute_threshold":.0005,"relative_threshold":.02,"effective_loss_threshold":threshold,
   "delta":row["G01_minus_F0"],"exceeds_loss_threshold":row["G01_minus_F0"]>threshold}
out={
 "schema":"independent_g01_A_tier_interim_score_summary_v1",
 "status":"interim_A_pair_only",
 "scope":"A2018/A2022 equally weighted descriptive retrospective development summary; both targets were exposed benchmarks. Not a prospective validation or a final G01 decision.",
 "inputs":[{"target":y,"path":str(INPUTS[str(y)]["path"]),"sha256":INPUTS[str(y)]["sha256"]} for y in (2018,2022)],
 "equal_cycle_means_all_races":means,
 "G01_minus_F0_equal_cycle_delta_all_races":delta,
 "primary_G01_minus_F0":primary,
 "practical_CRPS_review_aid":practical,
 "cycle_results":{
   str(c["target"]):{
    "all_race_G01_minus_F0":next(x for x in reports[c["target"]]["primary_G01_minus_F0"]["per_cycle"] if x["stratum"]=="all" and x["metric"]=="share_crps"),
    "competitive_G01_minus_F0":next(x for x in reports[c["target"]]["primary_G01_minus_F0"]["per_cycle"] if x["stratum"]=="competitive" and x["metric"]=="share_crps"),
    "F0_mask_counts":c["masks"],
    "near_even_status":"unavailable_empty_mask" if c["masks"]["near_even"]==0 else "available"
   } for c in cycles
 },
 "precision_limits":{
   "empty_near_even":"Both A-cycle F0-defined near-even masks are empty; all near-even A comparison scores and MCSE remain unavailable, not zero.",
   "independent_macro_cycles":"A has only two exposed national cycle units; equal weighting is descriptive and no race-count replication is implied.",
   "numerical_mcse":"Per-cycle delta MCSE uses hypot(G01,F0). A equal-cycle mean uses sqrt(sum(per-cycle delta MCSE^2))/2; numerical integration only."
 },
 "method":"Read only the previously created independent 2018-r2 and 2022 score reports. No prediction bank, outcome label, producer scoring function, model, or fit was opened.",
 "fit_or_resampling_invoked":False
}
(HERE/"independent-score-review-A-interim.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
print(json.dumps({"written":str(HERE/"independent-score-review-A-interim.json"),"all_race_crps_delta":delta["share_crps"],"competitive_mask":{str(c["target"]):c["masks"]["competitive"] for c in cycles},"near_even_empty":True}))
