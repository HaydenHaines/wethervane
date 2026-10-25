"""Receipt-gated independent output arithmetic; no scoring or forecast execution."""
import hashlib,json,math
from pathlib import Path
from datetime import datetime
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
OUT=BASE/'momentum-forecast-scoring-sol-v2/actual-root-v1'
METRICS=('crps','brier','negative_log','wis');ARMS=('FS5','FS6','LP5','F0_archived_predictive','PO5')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def checked(p):
 q=Path(p['path']);assert sha(q)==p['sha256'],str(q);return q
errors={};checks=0
def close(a,b,key,tol=2e-12):
 global checks
 checks+=1
 if a is None or b is None:assert a is None and b is None,(key,a,b);return
 e=abs(float(a)-float(b));errors[key]=max(errors.get(key,0.),e)
 assert e<=tol*max(1,abs(float(a)),abs(float(b))),(key,a,b,e)
def wis(y,q):
 total=.5*abs(y-q[.5])
 for alpha in (.5,.2,.05):
  lo,hi=q[alpha/2],q[1-alpha/2]
  total+=alpha/2*(hi-lo+2/alpha*max(lo-y,0)+2/alpha*max(y-hi,0))
 return total/3.5
terminal_path=OUT/'terminal.json'
if not terminal_path.exists():print('WAITING: no terminal');raise SystemExit(0)
terminal=json.loads(terminal_path.read_text())
if terminal.get('status')!='complete_scores':
 (HERE/'noncomplete-terminal.json').write_text(json.dumps(dict(terminal=pin(terminal_path),status=terminal.get('status')),indent=2)+'\n')
 print('NONCOMPLETE TERMINAL: numeric output not opened');raise SystemExit(2)
receipt_path=OUT/'receipt.json';receipt=json.loads(receipt_path.read_text())
assert terminal['receipt']==receipt
expected={'scores.json','cycle-influences.npz','cycle-influence-scales.json','launch.json','label-access.json'}
assert set(receipt['artifacts'])==expected
for value in receipt['artifacts'].values():checked(value)
# Only after every advertised output byte pin passes may numeric output be decoded.
scores=json.loads(checked(receipt['artifacts']['scores.json']).read_text())
scales=json.loads(checked(receipt['artifacts']['cycle-influence-scales.json']).read_text())
with np.load(checked(receipt['artifacts']['cycle-influences.npz']),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
assert set(arrays)==set(scales)
admission=json.loads(checked(receipt['admission']).read_text())
assert admission['scoring_manifest']['sha256']=='9cea17abf7b52a32f6817f210cab66e28f117a59a53b8d1032f2200c69c75eb1'
assert admission['independent_scoring_review']['sha256']=='aaa209723ee2e36da49645dfc19e7ce7128181358ef3e9eff6ee3879f0b75885'
runtime='01ff4314a314fcb315719c0efb30062d9ee81878fa12ae428f7f1645f56cd354'
assert admission['scoring_runtime']['sha256']==admission['runtime_receipt']['sha256']==admission['execution_identity']['runtime_sha256']==runtime
manifest=json.loads(checked(admission['scoring_manifest']).read_text());assert next(z for z in manifest['files'] if z['path']=='runtime.json')['sha256']==runtime
review=json.loads(checked(admission['independent_scoring_review']).read_text());assert review['runtime_sha256']==runtime
masks=json.loads(checked(admission['mask_receipt']).read_text())
assert set(scores['cycles'])=={'2018','2020','2022','2024'}
launch=json.loads(checked(receipt['artifacts']['launch.json']).read_text());access=json.loads(checked(receipt['artifacts']['label-access.json']).read_text())
assert launch['pid']==1603537
assert access['label_source']==admission['label_source']
assert datetime.fromisoformat(launch['utc'])<=datetime.fromisoformat(access['utc'])<=datetime.fromisoformat(terminal['utc'])
counts={};regressions=[];qualification=[];differences=[]
for cycle,cy in scores['cycles'].items():
 mask=masks['folds'][cycle];counts[cycle]={}
 assert cy['selected_attempt']==0
 for subset,part in cy['masks'].items():
  ids=mask[{'common':'common_polled_race_ids','competitive':'competitive_race_ids','near_even':'near_even_race_ids'}[subset]]
  assert part['race_ids']==ids;counts[cycle][subset]=len(ids)
  for arm,report in part['arms'].items():
   rows=report['rows'];assert [r['race_id'] for r in rows]==ids
   for row in rows:
    y=row['actual_dem_share'];assert 0<=y<=1 and row['tie']==(y==.5)
    q={float(k):v for k,v in row['quantiles'].items()}
    close(row['metrics']['wis']['value'],wis(y,q),'per_race_WIS')
    close(row['median_margin'],2*q[.5]-1,'median_margin')
    assert -1<=row['expected_margin']<=1
    for level,bounds in row['margin_intervals'].items():
     alpha=1-float(level);lo=min(q,key=lambda x:abs(x-alpha/2));hi=min(q,key=lambda x:abs(x-(1-alpha/2)))
     close(bounds[0],2*q[lo]-1,'margin_interval');close(bounds[1],2*q[hi]-1,'margin_interval')
    assert 0<=row['p_absolute_margin_le_02']<=1 and 0<=row['p_absolute_margin_ge_10']<=1
    pit=row['PIT'];assert pit['kind'] in ('continuous_CDF','discrete_CDF_bracket_no_randomization')
    if pit['kind']=='continuous_CDF':
     assert 0<=pit['value']<=1
     close(pit['value'],math.exp(pit['log_cdf']),'PIT_log_value')
     close(np.logaddexp(pit['log_cdf'],pit['log_survival']),0.,'PIT_complement')
    else:assert 0<=pit['lower']<=pit['upper']<=1;close(pit['midpoint'],(pit['lower']+pit['upper'])/2,'empirical_PIT_midpoint')
    if not row['tie']:
     winner=y>.5;p=row['win_probability']
     if arm=='F0_archived_predictive':
      brier=(p-int(winner))**2;true=p if winner else 1-p;loss=None if true==0 else -math.log(true)
     else:
      brier=math.exp(2*row['log_lose' if winner else 'log_win']);loss=-row['log_win' if winner else 'log_lose']
     close(row['metrics']['brier']['value'],brier,'per_race_Brier')
     close(row['metrics']['negative_log']['value'],loss,'per_race_NLL')
   for metric,agg in report['aggregates'].items():
    selected=[r for r in rows if not(r['tie'] and metric in ('brier','negative_log'))]
    assert agg['count']==len(selected)
    vals=[r['metrics'][metric]['value'] for r in selected]
    point=None if not selected or any(v is None for v in vals) else float(np.mean(vals))
    close(agg['value'],point,'race_to_cycle_'+metric)
    key=f'{cycle}__{subset}__{arm}__{metric}'
    if key in arrays:
     a=arrays[key];assert a.ndim==2 and a.shape[0]==4 and np.isfinite(a).all()
     c,d=a.shape;vars=[np.var(a.mean(axis=1),ddof=1)/c]
     for length in (50,100,200):
      assert d%length==0
      bm=a.reshape(c,d//length,length).mean(axis=2).reshape(-1)
      vars.append(np.var(bm,ddof=1)/len(bm))
     mcse=float(np.sqrt(max(vars)))
     if mcse>0 and agg.get('log_mcse') is not None:
      logerr=float(scales[key]+math.log(mcse));close(agg['log_mcse'],logerr,'influence_log_MCSE',5e-12)
      normal=math.exp(logerr);close(agg['mcse'],normal if normal else None,'influence_MCSE')
     elif mcse==0:assert not agg['qualified']
     nonlinear=agg.get('nonlinear_chain_check')
     if nonlinear is not None:
      residual=np.asarray(nonlinear['linearization_residual_scaled'],float)
      close(nonlinear['residual_log_scale'],scales[key],'nonlinear_log_scale')
      close(nonlinear['max_scaled_residual'],max(abs(residual)),'nonlinear_max_residual')
      close(nonlinear['scaled_limit'],3*mcse,'nonlinear_limit')
      assert nonlinear['passed']==bool(np.isfinite(residual).all() and max(abs(residual))<=3*mcse)
      # Raw nonlinear cycle points provide a distinct finite-precision check.
      raw=(np.asarray(nonlinear['scores'])-agg['value'])*math.exp(-scales[key])-a.mean(axis=1)
      for left,right in zip(raw,residual):close(left,right,'nonlinear_chain_point_linearization',2e-9)
     if agg['qualified']:
      assert agg['mcse'] is not None and agg['mcse']<=({'negative_log':.005}.get(metric,.0005))
      assert nonlinear is None or nonlinear['passed']
    qualification.append(dict(cycle=int(cycle),mask=subset,arm=arm,metric=metric,status=agg['status'],qualified=agg['qualified'],count=agg['count'],value=agg['value'],mcse=agg['mcse']))
   desc=report['calibration_and_width'];eligible=[r for r in rows if not r['tie']]
   assert sum(z['count'] for z in desc['win_calibration_bins'])==len(eligible)
   for z in desc['win_calibration_bins']:
    partrows=[r for r in eligible if z['lower']<=r['win_probability'] and (r['win_probability']<z['upper'] if z['upper']<1 else r['win_probability']<=1)]
    assert z['count']==len(partrows)
    close(z['mean_probability'],None if not partrows else np.mean([r['win_probability'] for r in partrows]),'calibration_bin_mean')
    close(z['observed_fraction'],None if not partrows else np.mean([r['actual_dem_share']>.5 for r in partrows]),'calibration_observed')
   for level,z in desc['interval_coverage_and_width'].items():
    alpha=1-float(level);cov=[];width=[]
    for row in rows:
     q={float(k):v for k,v in row['quantiles'].items()};lo=q[min(q,key=lambda x:abs(x-alpha/2))];hi=q[min(q,key=lambda x:abs(x-(1-alpha/2)))];cov.append(lo<=row['actual_dem_share']<=hi);width.append(hi-lo)
    assert z['count']==len(rows)
    close(z['coverage'],None if not rows else np.mean(cov),'interval_coverage');close(z['mean_width'],None if not rows else np.mean(width),'interval_width')
  for metric,diff in part['primary_FS6_minus_FS5'].items():
   a=part['arms']['FS6']['aggregates'][metric];bb=part['arms']['FS5']['aggregates'][metric]
   value=None if a['value'] is None or bb['value'] is None else a['value']-bb['value'];close(diff['value'],value,'cycle_difference')
   if diff['mcse'] is not None:close(diff['mcse'],math.hypot(a['mcse'],bb['mcse']),'cycle_difference_MCSE')
   if diff['mcse'] is not None:
    close(diff['three_mcse'],3*diff['mcse'],'difference_3MCSE')
    assert diff['numerically_resolved']==bool(value!=0 and abs(value)>=3*diff['mcse'])
   item=dict(cycle=int(cycle),mask=subset,metric=metric,value=value,mcse=diff['mcse'],resolved=diff['numerically_resolved'])
   differences.append(item)
   if value is not None and value>0:regressions.append(item)
for tier,years in [('A',('2018','2022')),('B',('2020','2024'))]:
 for subset,part in scores['tiers'][tier].items():
  assert part['cycles']==list(map(int,years)) and part['no_empty_cycle_drop']
  for arm,metrics in part['arms'].items():
   for metric,agg in metrics.items():
    cs=[scores['cycles'][y]['masks'][subset]['arms'][arm]['aggregates'][metric] for y in years]
    expected=None if any(z['value'] is None for z in cs) else np.mean([z['value'] for z in cs]);close(agg['value'],expected,'cycle_to_tier')
    assert agg['qualified']==all(z['qualified'] for z in cs)
    if agg['mcse'] is not None:close(agg['mcse'],math.hypot(cs[0]['mcse'],cs[1]['mcse'])/2,'tier_MCSE')
  for metric,diff in part['primary_FS6_minus_FS5'].items():
   a=part['arms']['FS6'][metric];bb=part['arms']['FS5'][metric]
   expected=None if a['value'] is None or bb['value'] is None else a['value']-bb['value'];close(diff['value'],expected,'tier_difference')
   if diff['mcse'] is not None:close(diff['mcse'],math.hypot(a['mcse'],bb['mcse']),'tier_difference_MCSE')
   assert diff==scores['primary_differences'][tier][subset][metric]
assert counts['2022']['near_even']==0
for arm in ARMS:
 for metric in METRICS:assert scores['tiers']['A']['near_even']['arms'][arm][metric]['value'] is None
summary=dict(schema='IndependentActualScoreAggregationAuditSolV1',terminal=pin(terminal_path),receipt=pin(receipt_path),artifacts=receipt['artifacts'],admission=receipt['admission'],checks=checks,max_errors=errors,mask_counts=counts,all_source_bound_aggregation_checks_passed=True,per_cycle_primary_differences=differences,all_positive_point_regressions=regressions,metric_qualification=qualification,scoring_or_forecast_rerun=False,actual_label_source_opened=False,full_count_or_promotion_inference=False)
(HERE/'aggregation-audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(checks=checks,max_error=max(errors.values(),default=0),counts=counts,regressions=len(regressions),terminal_sha256=sha(terminal_path)),indent=2))
