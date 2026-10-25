#!/usr/bin/env python3
"""Independent sealed-bank score audit. Never imports producer runner/scoring code."""
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np

LAB=Path("/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024")
RUN=LAB/"gasoline-execution-v1/attempt-v1"
EXEC_PLAN=LAB/"gasoline-execution-v1/prepared-v2/plan.json"
EXEC_PLAN_SHA="63f6de0e57949857a48d02e5084af8cc9d576b4ffa96985b13afc938fe55b5ab"
PREP_PLAN=LAB/"gasoline-experiment-v1/prepared-v2/plan.json"
PREP_PLAN_SHA="ee3cdefc9b3a5f9bc98598acd47ef7db81d0393f32c874396843e587566849ef"
ROOT_SELECTION=LAB/"checkpoints/gasoline-execution-root-selection-v1.json"
ROOT_SELECTION_SHA="e0a6728938cbe213c100fa71be512931abe9ed7c6291210bd0ba7b30ce164492"
ACTUAL_LAUNCH=LAB/"checkpoints/gasoline-actual-launch-v1.json"
ACTUAL_LAUNCH_SHA="a8b413a3dfb21e4c606e26abb8c9e28e5a1d0331bf442ec0a4eb74399608e4ef"
ARMS=("H0","F0","G01"); YEARS=(2018,2020,2022,2024)
FIELDS=("share_crps","share_wis","win_brier","win_log_score","margin_mae_pp","coverage80","width80","coverage95","width95")
TOL=2e-10; PINS={}

def digest(data): return hashlib.sha256(data).hexdigest()
def pin_path(path, expected=None):
    path=Path(path).resolve(); data=path.read_bytes(); got=digest(data)
    if expected is not None: assert got==expected, f"pin mismatch: {path}"
    PINS[str(path)]=got
    return data
def read_json(path, expected=None): return json.loads(pin_path(path,expected))
def checked(item): return pin_path(item["path"],item["sha256"])
def close(a,b,label):
    if isinstance(a,dict) or isinstance(b,dict):
        assert isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b),label
        for k in a: close(a[k],b[k],f"{label}.{k}")
    elif isinstance(a,str) or isinstance(b,str): assert a==b,f"{label}: {a!r} != {b!r}"
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),label
        for i,(x,y) in enumerate(zip(a,b)): close(x,y,f"{label}[{i}]")
    elif isinstance(a,bool) or isinstance(b,bool): assert a is b,f"{label}: {a!r} != {b!r}"
    elif a is None or b is None: assert a is b,f"{label}: {a!r} != {b!r}"
    else: assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=TOL),f"{label}: {a} != {b}"

def global_metadata():
    selected=read_json(ROOT_SELECTION,ROOT_SELECTION_SHA)
    assert selected["plan"]["sha256"]==EXEC_PLAN_SHA and selected["attempt"]==str(RUN)
    cp=read_json(ACTUAL_LAUNCH,ACTUAL_LAUNCH_SHA)
    assert cp["execution_plan_sha256"]==EXEC_PLAN_SHA and cp["selection_sha256"]==ROOT_SELECTION_SHA and cp["output"]==str(RUN)
    launch=read_json(RUN/"launch.json",cp["native_launch"]["sha256"])
    assert launch["plan_sha256"]==EXEC_PLAN_SHA and launch["plan"]["sha256"]==EXEC_PLAN_SHA
    execution=read_json(EXEC_PLAN,EXEC_PLAN_SHA); prep=read_json(PREP_PLAN,PREP_PLAN_SHA)
    assert execution["preparation"]["sha256"]==PREP_PLAN_SHA and len(execution["jobs"])==12
    assert execution["fit_count"]==12 and execution["promotion_authorized"] is False
    outcome_pin=next(x for x in prep["source_pins"] if x["path"].endswith("outcomes-SEPARATE.csv"))
    return execution,prep,outcome_pin

def preflight_fold(target,prep,execution):
    """Read seal/mask metadata only. No banks or labels until all selected folds pass this phase."""
    folder=RUN/str(target); fold=next(x for x in prep["folds"] if x["target_cycle"]==target)
    ids=fold["target_race_ids"]; seals={}; seal_pins=[]
    for arm in ARMS:
        p=folder/arm/"prediction-seal.json"; seal=read_json(p)
        assert (seal["target"],seal["arm"],seal["plan_sha256"])==(target,arm,EXEC_PLAN_SHA)
        assert seal["numerical_passed"] is True
        assert Path(seal["predictions"]["path"]).is_file()
        seals[arm]=seal; seal_pins.append({"path":str(p.resolve()),"sha256":digest(p.read_bytes())})
    mp=folder/"masks.json"; mask=read_json(mp)
    assert mask["source_prediction_seals"]==seal_pins
    assert mask["race_ids"]["all"]==ids and mask["policy"]==execution["score_policy"]
    terminals={}
    for arm in ARMS:
        t=read_json(folder/arm/"terminal.json")
        assert t["status"]=="completed" and t["numerical_passed"] is True and t["plan_sha256"]==EXEC_PLAN_SHA
        assert not t["forecast_qualified"] and not t["promotion_authorized"]
        terminals[arm]=t
    return fold,ids,seals,mask,terminals

def read_bank(seal,target,arm,ids):
    assert (seal["target"],seal["arm"],seal["plan_sha256"])==(target,arm,EXEC_PLAN_SHA)
    item=seal["predictions"]; raw=checked(item)
    assert item["path"].endswith(f"/{target}/{arm}/predictions.npz")
    with np.load(item["path"],allow_pickle=False) as z:
        assert {"shares","race_ids","posterior_draw","replicate"} <= set(z.files)
        shares=np.asarray(z["shares"],float); races=z["race_ids"].astype(str).tolist()
        assert races==ids and shares.shape==(4,20000,len(ids))
        assert np.isfinite(shares).all() and ((shares>=0)&(shares<=1)).all()
        assert np.array_equal(z["posterior_draw"],np.repeat(np.arange(4000),5))
        assert np.array_equal(z["replicate"],np.tile(np.arange(5),4000))
    assert digest(raw)==item["sha256"]
    return shares

def crps(flat,truth):
    ordered=np.sort(flat,axis=0,kind="stable"); n=len(ordered)
    coeff=2*np.arange(1,n+1)-n-1
    return np.abs(flat-truth[None,:]).mean(axis=0)-(coeff[:,None]*ordered).sum(axis=0)/(n*n)

def crps_influence(shares,truth):
    flat=shares.reshape(-1,shares.shape[-1]); order=np.argsort(flat,axis=0)
    ordered=np.take_along_axis(flat,order,axis=0); n=len(flat)
    prefix=np.cumsum(ordered,axis=0)-ordered
    pair_sorted=((2*np.arange(n)[:,None]-n)*ordered+ordered.sum(axis=0)-2*prefix)/n
    pair=np.empty_like(flat); np.put_along_axis(pair,order,pair_sorted,axis=0)
    return (np.abs(flat-truth[None,:])-pair).reshape(shares.shape)

def mcse_influence(x):
    c,n=x.shape; assert c==4 and n==20000 and n%1000==0
    variances=[np.var(x.mean(axis=1),ddof=1)/c]
    for size in (250,500):
        batch=x.reshape(c,n//size,size).mean(axis=2).reshape(-1)
        variances.append(np.var(batch,ddof=1)/len(batch))
    return np.sqrt(np.max(np.stack(variances),axis=0))

def score_mcse(shares,truth):
    p=(shares>.5).mean(axis=(0,1)); indicator=(shares>.5).astype(float); centered=indicator-p[None,None,:]
    win=(truth>.5).astype(float); clipped=np.clip(p,1e-6,1-1e-6)
    derivative=((1-win)/(1-clipped)-win/clipped)*((p>1e-6)&(p<1-1e-6))
    inf=[crps_influence(shares,truth).mean(axis=2),(2*(p-win)*centered).mean(axis=2),(derivative*centered).mean(axis=2)]
    return dict(zip(("share_crps","win_brier","win_log_score"),[float(mcse_influence(v)) for v in inf])),[i for i,x in enumerate(p) if x in (0,1)]

def score_rows(shares,truth,ids):
    flat=shares.reshape(-1,len(ids)); p=(shares>.5).mean(axis=(0,1)); win=truth>.5
    q025,q10,q50,q90,q975=np.quantile(flat,[.025,.1,.5,.9,.975],axis=0,method="linear")
    scores=crps(flat,truth); brier=(p-win.astype(float))**2
    log=-np.log(np.clip(np.where(win,p,1-p),1e-6,1.0))
    wis=.5*np.abs(q50-truth)
    for alpha,lo,hi in ((.2,q10,q90),(.05,q025,q975)):
        wis+=(alpha/2)*((hi-lo)+(2/alpha)*(np.maximum(lo-truth,0)+np.maximum(truth-hi,0)))
    wis/=2.5; mean=flat.mean(axis=0)
    rows=[]
    for j,race in enumerate(ids):
        pj=float(p[j]); y=float(truth[j])
        rows.append({
          "race_id":race,"actual_dem_share":y,"mean_dem_share":float(mean[j]),"share_crps":float(scores[j]),
          "win_probability":pj,"win_brier":float(brier[j]),"win_log_score":float(log[j]),
          "margin_mae_pp":float(200*abs(mean[j]-y)),
          "zero_empirical_true_outcome":int((pj if win[j] else 1-pj)==0),
          "winner_tail_precision":"unresolved_constant_empirical_indicator" if pj in (0,1) else "estimated_by_chain_batch_mcse",
          "coverage80":bool(q10[j]<=y<=q90[j]),"width80":float(q90[j]-q10[j]),
          "interval_score80":float((q90[j]-q10[j])+10*(max(q10[j]-y,0)+max(y-q90[j],0))),
          "coverage95":bool(q025[j]<=y<=q975[j]),"width95":float(q975[j]-q025[j]),
          "interval_score95":float((q975[j]-q025[j])+40*(max(q025[j]-y,0)+max(y-q975[j],0))),
          "share_wis":float(wis[j]),"expected_margin":float(2*mean[j]-1),"median_margin":float(2*q50[j]-1),
          "narrow_margin_probability":float((np.abs(2*flat[:,j]-1)<=.02).mean()),
          "large_D_margin_probability":float((2*flat[:,j]-1>=.1).mean()),
          "large_R_margin_probability":float((2*flat[:,j]-1<=-.1).mean())
        })
    return rows

def summarize(rows):
    if not rows:return {"status":"unavailable_empty_mask","n":0}
    return {"status":"available","n":len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in FIELDS}}

def truth_after_all_masks(outcome_pin,ids,target):
    pin_path(outcome_pin["path"],outcome_pin["sha256"])
    by_id={}
    with Path(outcome_pin["path"]).open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["cycle"])==target and row["race_id"] in ids:by_id[row["race_id"]]=float(row["actual_dem_share"])
    assert set(by_id)==set(ids)
    y=np.asarray([by_id[r] for r in ids],float)
    assert np.isfinite(y).all() and ((y>0)&(y<1)).all() and (y!=.5).all()
    return y

def refs_after_all_masks(outcome_pin,target,truth):
    by_cycle={}
    with Path(outcome_pin["path"]).open(newline="") as f:
        for row in csv.DictReader(f):
            c=int(row["cycle"])
            if 2002<=c<target:by_cycle.setdefault(c,[]).append(float(row["actual_dem_share"])>.5)
    assert by_cycle
    freq=float(np.mean([np.mean(v) for _,v in sorted(by_cycle.items())]))
    win=truth>.5
    return {"always_half_winner_brier":.25,"always_half_winner_log_score":math.log(2),
      "earlier_equal_cycle_D_win_frequency":freq,
      "earlier_frequency_brier":float(np.mean((freq-win.astype(float))**2)),
      "earlier_frequency_log_score":float(np.mean(-np.log(np.where(win,freq,1-freq))))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--go",action="store_true",help="Use only after root explicitly notifies fold-ready/completion")
    ap.add_argument("--target",type=int,choices=YEARS,action="append")
    ap.add_argument("--revision",type=int,default=1,help="Immutable report revision; increment instead of overwriting")
    args=ap.parse_args()
    if not args.go:
        print("Prepared only; pass --go after root notification. This mode reads no prediction banks or labels.")
        return
    execution,prep,outcome_pin=global_metadata(); targets=tuple(args.target or YEARS)
    # First validate seals/mask metadata for every requested fold. Then read banks and independently derive F0 masks.
    ready={}
    for year in targets:ready[year]=preflight_fold(year,prep,execution)
    prepared={}
    for year,(fold,ids,seals,mask,terminals) in ready.items():
        banks={arm:read_bank(seals[arm],year,arm,ids) for arm in ARMS}
        p=(banks["F0"]>.5).mean(axis=(0,1))
        derived={"all":list(ids),"competitive":[r for r,x in zip(ids,p) if .1<=x<=.9],
          "near_even":[r for r,x in zip(ids,p) if .4<=x<=.6]}
        assert mask["race_ids"]==derived
        prepared[year]=(fold,ids,seals,mask,terminals,banks)
    # All requested folds now have three valid sealed banks and verified F0 masks. Only now read saved labels.
    cycle_reports=[]
    for year in targets:
        fold,ids,seals,mask,terminals,banks=prepared[year]
        truth=truth_after_all_masks(outcome_pin,ids,year)
        refs=refs_after_all_masks(outcome_pin,year,truth)
        arm_report={}
        for arm in ARMS:
            rows=score_rows(banks[arm],truth,ids); strata={}
            for name,members in mask["race_ids"].items():
                indices=[ids.index(r) for r in members]; selected=[rows[i] for i in indices]
                summary=summarize(selected)
                if indices:
                    mc,unresolved=score_mcse(banks[arm][:,:,indices],truth[indices])
                    summary["mcse"]=mc
                    summary["unresolved_win_tail_race_ids"]=[members[i] for i in unresolved]
                    summary["tail_precision_resolved"]=not unresolved
                strata[name]=summary
            arm_report[arm]={"strata":strata,"references":refs,"numerical_passed":terminals[arm]["numerical_passed"]}
            score_path=RUN/str(year)/arm/"scores.json"
            saved=read_json(score_path)
            assert saved["numerical_passed"]==terminals[arm]["numerical_passed"]
            close(saved["references"],refs,f"{year}/{arm}/references")
            assert len(saved["rows"])==len(rows)
            for expected,got in zip(rows,saved["rows"]):
                assert expected["race_id"]==got["race_id"]
                for k,v in expected.items():
                    if k in got:close(v,got[k],f"{year}/{arm}/{expected['race_id']}/{k}")
            for name,summary in strata.items():
                stored=saved["strata"][name]
                if summary["n"]==0:assert stored["status"]=="unavailable_empty_mask";continue
                for k,v in summary.items():
                    if k in stored:close(v,stored[k],f"{year}/{arm}/{name}/{k}")
            arm_report[arm]=arm_report[arm]
        cycle_reports.append({"target":year,"tier":fold["tier"],"race_count":len(ids),
          "masks":{k:len(v) for k,v in mask["race_ids"].items()},"arms":arm_report})
    tiers={}; per_cycle=[]; equal_cycle=[]
    for c in cycle_reports:
        for stratum in ("all","competitive","near_even"):
            for metric in ("share_crps","win_brier","win_log_score"):
                f0,g01=(c["arms"][arm]["strata"][stratum] for arm in ("F0","G01"))
                if not f0["n"] or not g01["n"]:
                    per_cycle.append({"target":c["target"],"tier":c["tier"],"stratum":stratum,
                      "metric":metric,"status":"unavailable_empty_mask","G01_minus_F0":None,
                      "combined_mcse":None,"loss":None,"loss_beyond_two_mcse":None,"precision_resolved":False})
                    continue
                delta=g01[metric]-f0[metric]; err=math.hypot(g01["mcse"][metric],f0["mcse"][metric])
                resolved=(metric=="share_crps" or (f0["tail_precision_resolved"] and g01["tail_precision_resolved"]))
                per_cycle.append({"target":c["target"],"tier":c["tier"],"stratum":stratum,"metric":metric,
                  "status":"available","G01_minus_F0":delta,"combined_mcse":err,"loss":delta>0,
                  "loss_beyond_two_mcse":delta>2*err if resolved else None,"precision_resolved":resolved})
    for tier,years in (("A",(2018,2022)),("B",(2020,2024))):
        selected=[c for c in cycle_reports if c["tier"]==tier]
        if len(selected)!=2:
            tiers[tier]={"status":"unavailable_incomplete_cycle_pair","targets":[c["target"] for c in selected]};continue
        means={}
        for arm in ARMS:
            means[arm]={k:float(np.mean([c["arms"][arm]["strata"]["all"][k] for c in selected])) for k in FIELDS}
        tiers[tier]={"status":"available","targets":[c["target"] for c in selected],"equal_cycle_means":means,
          "G01_minus_F0":{k:means["G01"][k]-means["F0"][k] for k in FIELDS}}
        for stratum in ("all","competitive","near_even"):
            for metric in ("share_crps","win_brier","win_log_score"):
                parts=[]
                for c in selected:
                    f0,g01=(c["arms"][a]["strata"][stratum] for a in ("F0","G01"))
                    if not f0["n"] or not g01["n"]:parts=[];break
                    delta=g01[metric]-f0[metric]; err=math.hypot(g01["mcse"][metric],f0["mcse"][metric])
                    resolved=(metric=="share_crps" or (f0["tail_precision_resolved"] and g01["tail_precision_resolved"]))
                    parts.append((c["target"],delta,err,resolved))
                if len(parts)==2:
                    delta=float(np.mean([x[1] for x in parts])); err=float(math.sqrt(sum(x[2]**2 for x in parts))/2)
                    resolved=all(x[3] for x in parts)
                    equal_cycle.append({"tier":tier,"stratum":stratum,"metric":metric,"G01_minus_F0":delta,
                      "combined_mcse":err,"loss":delta>0,"loss_beyond_two_mcse":delta>2*err if resolved else None,
                      "precision_resolved":resolved})
    report={"schema":"independent_g01_score_arithmetic_review_v1","status":"independent_recomputation_complete",
      "scope":"Retrospective exposed development comparison only; no fit, resampling, qualification, or promotion.",
      "plan_sha256":EXEC_PLAN_SHA,"preparation_sha256":PREP_PLAN_SHA,"selection_sha256":ROOT_SELECTION_SHA,
      "per_cycle":cycle_reports,"A_B_equal_cycle_summaries":tiers,
      "primary_G01_minus_F0":{"per_cycle":per_cycle,"equal_cycle_tiers":equal_cycle},
      "precision_limits":{
        "constant_probability":"Constant empirical winner tails stay explicitly unresolved even if computed MCSE is zero.",
        "empty_near_even":"Empty masks remain unavailable with n=0 and no score/MCSE; never zero-filled.",
        "replication":"Equal-cycle summaries; race rows are not independent national macro replications.",
        "numerical_error":"Per-cycle delta MCSE=hypot(G01,F0); two-cycle equal-cycle mean MCSE=sqrt(sum(per-cycle MCSE^2))/2; numerical integration only."
      },
      "source_pins":dict(PINS),"fit_or_sampler_invoked":False,"producer_scores_compared_after_recalculation":True}
    dirname=Path(__file__).resolve().parent
    tag=str(targets[0]) if len(targets)==1 else "all" if tuple(targets)==YEARS else "-".join(map(str,targets))
    rev=f"-r{args.revision}" if args.revision>1 else ""
    out=dirname/f"independent-score-review-{tag}{rev}.json"
    with out.open("x") as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"written":str(out),"folds":targets,"fit_or_sampler_invoked":False}))
if __name__=="__main__":main()
