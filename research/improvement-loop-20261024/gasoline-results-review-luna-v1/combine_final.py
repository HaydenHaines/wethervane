#!/usr/bin/env python3
"""Combine pinned per-cycle independent reports and compare to completed producer summary; no raw reread."""
import hashlib,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
LAB=HERE.parent
RUN=LAB/"gasoline-execution-v1/attempt-v1"
INPUTS={
 2018:("independent-score-review-2018-r2.json","c320ee29350de4e077a2d39bb9c16862757c330d37c53629444a8f313350eaef"),
 2020:("independent-score-review-2020.json","5bb744100205bfc64e45d86b0b562273fc93f25c69f560b3a199b18e7338419f"),
 2022:("independent-score-review-2022.json","8e2f3ee714cbfda0b94b0f6038b6e582e6c4627ccf41e5ca695d9bef564a9e43"),
 2024:("independent-score-review-2024.json","c56e14fa614e4efc6976057aa9cb950d2a30bbe63244a9ad72be9314fe56e3d8"),
}
TERMINAL_SHA="f9922ed230ce99c83f88d8663f7ec1f3b97634e0707e4dd62ac1f91fc4ac4ec9"
SUMMARY_SHA="b72223532e63555332a3628ab4b9d4aeafd5a3f2d926f6111f126f7e8e673b22"
ARMS=("H0","F0","G01")
FIELDS=("share_crps","share_wis","win_brier","win_log_score","margin_mae_pp","coverage80","width80","coverage95","width95")
PRIMARY=("share_crps","win_brier","win_log_score")
pins={}
def read(path,expected=None):
 b=Path(path).read_bytes();h=hashlib.sha256(b).hexdigest()
 if expected is not None: assert h==expected,(str(path),h,expected)
 pins[str(Path(path).resolve())]=h
 return json.loads(b)
def close(a,b,where,tol=2e-10):
 if isinstance(a,dict) or isinstance(b,dict):
  assert isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b),where
  for k in a:close(a[k],b[k],where+"."+k,tol)
 elif isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):
  assert len(a)==len(b),where
  for i,(x,y) in enumerate(zip(a,b)):close(x,y,f"{where}[{i}]",tol)
 elif isinstance(a,bool) or isinstance(b,bool):assert a is b,where
 elif a is None or b is None:assert a is b,where
 elif isinstance(a,str) or isinstance(b,str):assert a==b,where
 else:assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=tol),(where,a,b)

reports={}
for year,(name,digest) in INPUTS.items():
 d=read(HERE/name,digest)
 assert d["status"]=="independent_recomputation_complete" and d["fit_or_sampler_invoked"] is False
 assert len(d["per_cycle"])==1 and d["per_cycle"][0]["target"]==year
 reports[year]=d
terminal=read(RUN/"terminal.json",TERMINAL_SHA)
summary=read(RUN/"summary.json",SUMMARY_SHA)
assert terminal["status"]=="completed_F_only_comparison"
assert terminal["all_numerical_passed"] is True and terminal["summary"]["sha256"]==SUMMARY_SHA
assert not terminal["count_qualification"] and not terminal["promotion_authorized"]
assert not summary["count_qualification"] and not summary["promotion_authorized"] and summary["prospective_evidence"] is False
# Compare every saved cycle/arm/stratum metric and its availability/precision state with independent per-cycle report.
findings=[]
for cycle in summary["cycles"]:
 year=cycle["target"]; independent=reports[year]["per_cycle"][0]
 assert cycle["tier"]==independent["tier"] and cycle["tier"]==("A" if year in (2018,2022) else "B")
 for arm in ARMS:
  actual=cycle["arms"][arm]; wanted=independent["arms"][arm]
  assert actual["numerical_passed"]==wanted["numerical_passed"] is True
  for stratum in ("all","competitive","near_even"):
   got=actual["strata"][stratum]; expected=wanted["strata"][stratum]
   assert set(got).issubset(expected),f"{year}/{arm}/{stratum} unexpected stored fields"
   for name,value in got.items():close(value,expected[name],f"{year}/{arm}/{stratum}/{name}")
   if got["n"]:
    assert expected.get("tail_precision_resolved")== (not bool(got.get("unresolved_win_tail_race_ids",[])))
   else:
    assert got["status"]==expected["status"]=="unavailable_empty_mask"
 assert cycle["target"]==year
 findings.append({"target":year,"tier":cycle["tier"],"all_three_arms_numerical_passed":True,
  "mask_counts":independent["masks"],"near_even_status":"unavailable_empty_mask" if not independent["masks"]["near_even"] else "available"})
# Independently aggregate A/B means and compare the full producer equal-cycle summary.
tiers={}
for tier,years in (("A",(2018,2022)),("B",(2020,2024))):
 cycles=[reports[y]["per_cycle"][0] for y in years]
 armmeans={}
 for arm in ARMS:
  armmeans[arm]={m:float(np.mean([c["arms"][arm]["strata"]["all"][m] for c in cycles])) for m in FIELDS}
 deltas={m:armmeans["G01"][m]-armmeans["F0"][m] for m in FIELDS}
 ptiers=[]
 for stratum in ("all","competitive","near_even"):
  for metric in PRIMARY:
   parts=[]
   for c in cycles:
    f0,g=(c["arms"][a]["strata"][stratum] for a in ("F0","G01"))
    if not f0["n"] or not g["n"]:
     parts=[];break
    primary=next(x for x in reports[c["target"]]["primary_G01_minus_F0"]["per_cycle"]
      if x["stratum"]==stratum and x["metric"]==metric)
    parts.append(primary)
   if len(parts)!=2:
    ptiers.append({"tier":tier,"stratum":stratum,"metric":metric,
      "status":"unavailable_missing_or_empty_cycle_mask","equal_cycle_delta":None,"numerical_mcse":None,
      "loss":None,"tail_precision_resolved":False,"loss_beyond_two_mcse":None})
    continue
   delta=float(np.mean([x["G01_minus_F0"] for x in parts]))
   err=float(math.sqrt(sum(x["combined_mcse"]**2 for x in parts))/2)
   resolved=all(x["precision_resolved"] for x in parts)
   ptiers.append({"tier":tier,"stratum":stratum,"metric":metric,"status":"available",
     "equal_cycle_delta":delta,"numerical_mcse":err,"loss":delta>0,"tail_precision_resolved":resolved,
     "loss_beyond_two_mcse":delta>2*err if resolved else None})
 tiers[tier]={"cycles":list(years),"equal_cycle_means":armmeans,"G01_minus_F0":deltas,"primary_comparison":ptiers}
 # Compare all arm tier metrics and deltas exactly within tolerance.
 producer=summary["tiers"][tier]
 assert producer["cycles"]==list(years)
 for arm in ARMS:
  close(producer["equal_cycle_means"][arm],{k:armmeans[arm][k] for k in producer["equal_cycle_means"][arm]},f"tier {tier} means/{arm}")
 close(producer["G01_minus_F0"],{k:deltas[k] for k in producer["G01_minus_F0"]},f"tier {tier} deltas")
# Build primary regression records and cross-check every producer loss/precision flag.
per_cycle=[];equal_cycle=[]
for year in (2018,2020,2022,2024):
 per_cycle.extend(reports[year]["primary_G01_minus_F0"]["per_cycle"])
for tier in ("A","B"):
 expected=tiers[tier]["primary_comparison"]
 for row in expected:
  if row["status"]!="available":
   equal_cycle.append({"tier":tier,"stratum":row["stratum"],"metric":row["metric"],
     "status":"unavailable_missing_or_empty_cycle_mask","G01_minus_F0":None,"combined_mcse":None,
     "loss":None,"loss_beyond_two_mcse":None,"precision_resolved":False})
   continue
  years=tiers[tier]["cycles"]
  cycle_rows=[next(x for x in reports[y]["primary_G01_minus_F0"]["per_cycle"]
       if x["stratum"]==row["stratum"] and x["metric"]==row["metric"]) for y in years]
  equal_cycle.append({"tier":tier,"stratum":row["stratum"],"metric":row["metric"],"status":"available",
    "G01_minus_F0":row["equal_cycle_delta"],"combined_mcse":row["numerical_mcse"],
    "loss":row["loss"],"loss_beyond_two_mcse":row["loss_beyond_two_mcse"],
    "precision_resolved":row["tail_precision_resolved"]})
actual_reg=summary["regressions"]
wanted_per=[]
for x in per_cycle:
 if x["status"]!="available": continue
 metric=x["metric"]
 baseline=reports[x["target"]]["per_cycle"][0]["arms"]["F0"]["strata"][x["stratum"]]
 threshold=max(.0005,.02*baseline["share_crps"]) if metric=="share_crps" else None
 wanted_per.append({
  "target":x["target"],"tier":x["tier"],"arm":"G01","stratum":x["stratum"],"metric":metric,
  "challenger_minus_F0":x["G01_minus_F0"],"numerical_mcse":x["combined_mcse"],"loss":x["loss"],
  "tail_precision_resolved":x["precision_resolved"],"loss_beyond_two_mcse":x["loss_beyond_two_mcse"],
  "practical_CRPS_aid":metric=="share_crps" and x["G01_minus_F0"]>threshold
 })
def key(r):return (r["target"],r["tier"],r["arm"],r["stratum"],r["metric"])
actual_by={key(x):x for x in actual_reg["per_cycle"]}; wanted_by={key(x):x for x in wanted_per}
assert set(actual_by)==set(wanted_by)
for k in actual_by:close(actual_by[k],wanted_by[k],f"per-cycle regression {k}")
actual_tiers={(x["tier"],x["stratum"],x["metric"]):x for x in actual_reg["equal_cycle_tiers"]}
wanted_tiers={(x["tier"],x["stratum"],x["metric"]):x for x in equal_cycle}
assert set(actual_tiers)==set(wanted_tiers)
for k in actual_tiers:
 x=actual_tiers[k]; y=wanted_tiers[k]
 # Runner uses different field names: status/available, equal_cycle_delta/numerical_mcse, tail precision.
 close(x["status"],y["status"],f"tier regression status {k}")
 if x["status"]!="available":
  # Producer records only the unavailability status for empty/missing masks.
  continue
 for f1,f2 in (("equal_cycle_delta","G01_minus_F0"),("numerical_mcse","combined_mcse"),
               ("loss","loss"),("tail_precision_resolved","precision_resolved"),
               ("loss_beyond_two_mcse","loss_beyond_two_mcse")):
  close(x.get(f1),y.get(f2),f"tier regression {k}/{f1}")
# Keep actual B cycle comparisons beside equal-cycle result for the requested tradeoff review.
b_cycle=[]
for year in (2020,2024):
 r=next(c for c in reports[year]["primary_G01_minus_F0"]["per_cycle"] if c["stratum"]=="all")
 b_cycle.append(r)
# A practical CRPS aid, preserving its threshold and no-adoption status.
practical={}
for tier in ("A","B"):
 f0=tiers[tier]["equal_cycle_means"]["F0"]["share_crps"]
 d=tiers[tier]["G01_minus_F0"]["share_crps"]
 threshold=max(.0005,.02*f0)
 practical[tier]={"threshold":threshold,"delta":d,"exceeds_loss_threshold":d>threshold}
out={
 "schema":"independent_g01_final_score_arithmetic_review_v1",
 "status":"all_four_fold_recomputation_and_producer_summary_comparison_passed",
 "scope":"Exposed retrospective development evidence only; not prospective validation or full count qualification.",
 "per_cycle_independent_reports":[{"target":y,"path":INPUTS[y][0],"sha256":INPUTS[y][1]} for y in (2018,2020,2022,2024)],
 "producer_terminal":{"path":str((RUN/"terminal.json").resolve()),"sha256":TERMINAL_SHA},
 "producer_summary":{"path":str((RUN/"summary.json").resolve()),"sha256":SUMMARY_SHA},
 "per_cycle_masks_numerical_passes":findings,
 "equal_cycle_tiers":tiers,
 "primary_comparison_per_cycle":per_cycle,
 "primary_comparison_equal_cycle":equal_cycle,
 "B_cycle_primary_results":b_cycle,
 "practical_CRPS_review_aid":practical,
 "comparison_checks":{
   "cycle_arm_strata_metrics_masks_and_MCSE_flags":"matched independent reports",
   "A_B_equal_cycle_metrics":"matched producer summary",
   "per_cycle_delta_and_regression_flags":"matched producer regression report",
   "equal_cycle_delta_MCSE_and_flags":"matched producer regression report",
   "no_fill_for_empty_masks":True,
   "all_numerical_passed":terminal["all_numerical_passed"],
   "promotion_authorized":False
 },
 "assessment":{
   "recommendation":"retain F0; do not adopt G01 from these development comparisons",
   "A_tier":"All-race share CRPS improves by about 0.000150 equal-cycle, but this averages a 2018 gain against a 2022 loss. A-tier win Brier and log score worsen, and competitive-race metrics are materially worse; both A near-even masks are empty.",
   "B_tier":"All-race share CRPS is slightly better on equal-cycle average, with a 2020 loss and 2024 gain. Win Brier is worse in both 2020 and 2024; win log score improves in 2020 but worsens in 2024. This does not establish stable added value.",
   "limits":"Each target was an already exposed development benchmark. Numerical MCSE measures integration precision only; four election cycles are not untouched validation, and state-race rows do not multiply national feature replication."
 },
 "precision_limits":{
   "constant_probability":"Constant empirical winner tails remain unresolved even when the empirical influence MCSE is numerically zero; per-cycle/tier flags are retained.",
   "empty_near_even":"A near-even is unavailable in both A cycles; B near-even uses only available fixed-F0 masks and is not zero-filled.",
   "numerical_error":"Independent-arm delta MCSE is hypot(G01,F0); equal-cycle mean MCSE is sqrt(sum(per-cycle delta MCSE^2))/2. Numerical integration only."
 },
 "fit_resampling_or_reread_of_prior_banks":False,
 "source_pins":pins
}
output=HERE/"independent-score-review-all4.json"
with output.open("x") as stream:stream.write(json.dumps(out,indent=2,allow_nan=False)+"\n")
print(json.dumps({"written":str(output),"all_numerical_passed":terminal["all_numerical_passed"],
 "A_crps_delta":tiers["A"]["G01_minus_F0"]["share_crps"],"B_crps_delta":tiers["B"]["G01_minus_F0"]["share_crps"],
 "reports_compared":4}))
