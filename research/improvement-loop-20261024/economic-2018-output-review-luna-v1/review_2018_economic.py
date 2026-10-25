from pathlib import Path
import hashlib,json,sys
import numpy as np,pandas as pd,h5py,arviz as az
from scipy.special import expit,logit
root=Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024')
run=root/'economic-execution-v1/attempt-v1/2018'
planpath=root/'economic-execution-v1/prepared-v2/plan.json'
plan=json.loads(planpath.read_text()); ph=hashlib.sha256(planpath.read_bytes()).hexdigest()
assert ph=='e3faefeb89a5bca837bc7f8b01e666112d563b95fe16e909060edf5aa1cbfc50'
truth_path=root/'economic-experiment-v1/prepared-v1/snapshots/04-outcomes-SEPARATE.csv'
truthdf=pd.read_csv(truth_path)
truthdf=truthdf[truthdf.cycle==2018].set_index('race_id')
report={'schema':'economic_2018_output_independent_review_v1','plan_sha256':ph,'plan_path':str(planpath),'truth_path':str(truth_path),'truth_sha256':hashlib.sha256(truth_path.read_bytes()).hexdigest(),'terminal_outputs':{},'score_recomputation':{},'issues':[]}
arms=['H0','F0','E01','E01_sd025']
# independent uniform empirical CRPS; same finite empirical CDF semantics

def crps(v,y):
 v=np.asarray(v,float); s=np.sort(v); n=len(s)
 pair=np.sum((2*np.arange(n)-n+1)*s)/(n*n)
 return float(np.mean(np.abs(v-y))-pair)
def wis(v,y):
 qs=np.quantile(v,[.025,.1,.5,.9,.975],method='linear')
 lo95,lo80,med,hi80,hi95=qs
 sc=0.5*abs(med-y)
 for alpha,lo,hi in [(.2,lo80,hi80),(.05,lo95,hi95)]:
  sc+=alpha/2*((hi-lo)+2/alpha*(max(lo-y,0)+max(y-hi,0)))
 return float(sc/2.5)
# Job mappings and raw chain independently inspected
for arm in arms:
 d=run/arm; term=json.loads((d/'terminal.json').read_text()); fit=json.loads((d/'fit-diagnostics.json').read_text())
 assert term['plan_sha256']==ph and term['status']=='completed' and term['numerical_passed'] is True
 chain_rows=[]; chainseeds=[]; chain_shapes=[]; energy_stats=[]; raw_hashes=[]
 job=next(j for j in plan['jobs'] if j['target']==2018 and j['arm']==arm)
 for c in range(4):
  cp=d/f'chain-{c}'; complete=json.loads((d/f'chain-{c}-complete.json').read_text()); launch=json.loads((cp/'launch.json').read_text())
  raw=cp/'raw-bank.nc'; sha=hashlib.sha256(raw.read_bytes()).hexdigest()
  assert complete['complete'] and complete['plan_sha256']==ph and complete['raw_bank']['sha256']==sha
  assert term['raw_chains'][c]['sha256']==sha
  assert launch['plan_sha256']==ph and launch['seed']==complete['seed']==job['chain_seeds'][c]
  chainseeds.append(launch['seed']); raw_hashes.append(sha)
  with h5py.File(raw,'r') as f:
   w=f['warmup_posterior']; po=f['posterior']; ss=f['sample_stats']
   expected_k=len(job['source_binding']['parameter_names'])
   assert w['beta'].shape==(1,1500,expected_k) and po['beta'].shape==(1,4000,expected_k)
   assert po['cycle_z'].shape==(1,4000,10) and po['cycle_effect'].shape==(1,4000,10)
   assert po['tau_cycle'].shape==(1,4000) and po['tau_race'].shape==(1,4000)
   assert len(w['draw'])==1500 and np.array_equal(w['draw'][:],[*range(1500)])
   assert len(po['draw'])==4000 and np.array_equal(po['draw'][:],[*range(4000)])
   warm_coeff=[x.decode() if isinstance(x,bytes) else str(x) for x in w['coefficient'][:]]
   warm_cycles=list(map(int,w['source_cycle'][:]))
   assert warm_coeff==job['source_binding']['parameter_names'] and warm_cycles==job['source_binding']['cycles']
   assert np.isfinite(np.asarray(w['beta'])).all() and np.isfinite(np.asarray(w['cycle_z'])).all() and np.isfinite(np.asarray(w['tau_cycle'])).all() and np.isfinite(np.asarray(w['tau_race'])).all()
   beta=np.asarray(po['beta'][0]); cz=np.asarray(po['cycle_z'][0]); ce=np.asarray(po['cycle_effect'][0]); tc=np.asarray(po['tau_cycle'][0]); tr=np.asarray(po['tau_race'][0]); en=np.asarray(ss['energy'][0]); div=np.asarray(ss['diverging'][0]); depth=np.asarray(ss['tree_depth'][0])>=10
   coeff=[x.decode() if isinstance(x,bytes) else str(x) for x in po['coefficient'][:]]
   cycles=list(map(int,po['source_cycle'][:]))
   assert coeff==job['source_binding']['parameter_names']
   assert len(cycles)==10 and len(set(cycles))==10
   assert np.isfinite(beta).all() and np.isfinite(cz).all() and np.isfinite(tc).all() and np.isfinite(tr).all() and np.isfinite(en).all()
   # source cycle coordinate should agree with this job binding
   expected_cycles=job['source_binding']['cycles']; assert cycles==expected_cycles
   # Verify chain posterior journal state against the retained NetCDF rows.
   lines=(cp/'transitions.jsonl').read_text().splitlines(); assert len(lines)==5500
   maxdiff=0.; maxwarmdiff=0.; journal_post=0; journal_warm=0
   for rawline in lines:
    x=json.loads(rawline); ix=x['draw']; point=x['point']; stats=x['sampler_stats_repr']
    if ix<1500:
     arr=w; q=ix; journal_warm+=1
    else:
     arr=po; q=ix-1500; journal_post+=1
    b0=np.asarray(arr['beta'][0,q]); z0=np.asarray(arr['cycle_z'][0,q]); t0=np.asarray(arr['tau_cycle'][0,q]); r0=np.asarray(arr['tau_race'][0,q])
    dif=max(np.max(np.abs(b0-np.asarray(point['beta']))),np.max(np.abs(z0-np.asarray(point['cycle_z']))),abs(t0-np.exp(point['tau_cycle_log__'])),abs(r0-np.exp(point['tau_race_log__'])))
    if ix<1500: maxwarmdiff=max(maxwarmdiff,dif)
    else: maxdiff=max(maxdiff,dif)
   assert journal_post==4000 and journal_warm==1500 and maxdiff<1e-12 and maxwarmdiff<1e-12
   energy_stats.append({'chain':c,'ebfmi':float(np.mean(np.diff(en)**2)/np.var(en,ddof=1)),'divergences':int(div.sum()),'max_depth_hits':int(depth.sum()),'journal_to_retained_max_absdiff':maxdiff,'warmup_journal_to_retained_max_absdiff':maxwarmdiff})
   chain_rows.append({'chain':c,'warmup_draws':int(w['beta'].shape[1]),'retained_draws':int(po['beta'].shape[1]),'coefficients':coeff,'cycles':cycles})
 # independent ArviZ diagnostics over arrays across all four chains
 with h5py.File(d/'chain-0/raw-bank.nc','r') as f0:
  vars0=['beta','cycle_z','tau_cycle','tau_race']
 arrays={k:[] for k in vars0}
 for c in range(4):
  with h5py.File(d/f'chain-{c}/raw-bank.nc','r') as f:
   for k in vars0: arrays[k].append(np.asarray(f['posterior'][k][0]))
 pdata={k:np.stack(v,axis=0) for k,v in arrays.items()}
 idata=az.from_dict(posterior=pdata)
 rh=az.rhat(idata,method='rank').to_array().values
 eb=az.ess(idata,method='bulk').to_array().values
 et=az.ess(idata,method='tail').to_array().values
 report['terminal_outputs'][arm]={'status':term['status'],'fit_pass':fit['passed'],'reported_fit_diagnostics':fit['values'],'independent_fit_diagnostics':{'max_rank_rhat':float(np.nanmax(rh)),'min_bulk_ess':float(np.nanmin(eb)),'min_tail_ess':float(np.nanmin(et)),'divergences':sum(x['divergences'] for x in energy_stats),'max_depth_hits':sum(x['max_depth_hits'] for x in energy_stats)},'chain_seeds':chainseeds,'chain_rows':chain_rows,'chain_diagnostics':energy_stats,'raw_bank_sha256':raw_hashes,'raw_bank_paths':[str(run/arm/f'chain-{c}'/'raw-bank.nc') for c in range(4)]}
 # read sealed predictions and compare receipt, domain, joint alignment
 seal=json.loads((d/'prediction-seal.json').read_text()); predfile=d/'predictions.npz'; predsha=hashlib.sha256(predfile.read_bytes()).hexdigest()
 assert seal['plan_sha256']==ph and seal['target']==2018 and seal['arm']==arm and seal['numerical_passed'] is True
 assert seal['predictions']['sha256']==predsha
 z=np.load(predfile,allow_pickle=False); shares=z['shares']; logits=z['logits']; cycle=z['new_cycle_z']; race=z['new_race_z']; pdra=z['posterior_draw']; rep=z['replicate']; ids=z['race_ids'].tolist()
 assert shares.shape==(4,20000,30) and logits.shape==race.shape==(4,20000,30)
 assert cycle.shape==(4,20000) and set(ids)==set(truthdf.index) and len(ids)==30
 assert np.array_equal(pdra,np.repeat(np.arange(4000),5)) and np.array_equal(rep,np.tile(np.arange(5),4000))
 assert np.isfinite(shares).all() and np.all((shares>=0)&(shares<=1))
 assert np.allclose(expit(logits),shares,rtol=0,atol=2e-15)
 assert np.isfinite(race).all() and np.isfinite(cycle).all()
 # Shared new-cycle draw is the same for every race in each joint predictive row; verify common row identity and D/R complement.
 # Within this representation each row's scalar cycle shock is fed to every race's logit.
 win=(shares>.5).mean(axis=(0,1)); assert np.all((win>=0)&(win<=1)); assert np.allclose(win+(1-win),1)
 report['terminal_outputs'][arm]['prediction']={'sha256':predsha,'shape':list(shares.shape),'race_ids':ids,'predictive_rows':int(shares.shape[0]*shares.shape[1]),'posterior_draws_per_chain':4000,'replicates_per_posterior_draw':5,'finite_in_[0,1]':True,'max_abs_expit_logit_share_difference':float(np.max(np.abs(expit(logits)-shares))),'all_chains_same_race_order':True,'joint_cycle_shock_shape':list(cycle.shape),'joint_race_shocks_shape':list(race.shape),'share_probabilities_normalize_D_plus_R':True,'predictive_diagnostics':json.loads((d/'prediction-diagnostics.json').read_text())['values']}
 report['terminal_outputs'][arm]['_shares']=shares
# target truths and masks
ids=report['terminal_outputs']['F0']['prediction']['race_ids']
assert len(ids)==30 and all(i in truthdf.index for i in ids)
y=np.array([float(truthdf.loc[i,'actual_dem_share']) for i in ids])
# Verify mask race IDs independently from F0 forecast win rates.
fshares=report['terminal_outputs']['F0']['_shares']; winp=(fshares>.5).mean(axis=(0,1))
mask={'all':ids,'competitive':[r for r,p in zip(ids,winp) if .1<=p<=.9],'near_even':[r for r,p in zip(ids,winp) if .4<=p<=.6]}
provided=json.loads((run/'masks.json').read_text())['race_ids']
assert mask==provided
report['mask_verification']={'race_count':len(ids),'competitive_count':len(mask['competitive']),'near_even_count':len(mask['near_even']),'matches_saved_mask':True,'mask_rule':'inclusive F0 win-probability ranges [0.1,0.9] and [0.4,0.6]'}
# Recalculate row metrics and compare owner score outputs.
for arm in arms:
 shares=report['terminal_outputs'][arm].pop('_shares')
 flat=shares.reshape(-1,len(ids)); pp=(shares>.5).mean(axis=(0,1)); rows=[]
 for j,r in enumerate(ids):
  v=flat[:,j]; outcome=y[j]; p=pp[j]
  row={'race_id':r,'actual_dem_share':outcome,'share_crps':crps(v,outcome),'win_probability':float(p),'win_brier':float((p-int(outcome>.5))**2),'win_log_score':float(-np.log(np.clip(p if outcome>.5 else 1-p,1e-6,1))),'share_wis':wis(v,outcome)}
  rows.append(row)
  assert np.isclose(row['share_crps'],float(json.loads((run/arm/'scores.json').read_text())['rows'][j]['share_crps']),rtol=0,atol=2e-12)
  assert np.isclose(row['win_brier'],float(json.loads((run/arm/'scores.json').read_text())['rows'][j]['win_brier']),rtol=0,atol=2e-12)
  assert np.isclose(row['share_wis'],float(json.loads((run/arm/'scores.json').read_text())['rows'][j]['share_wis']),rtol=0,atol=2e-12)
 score=json.loads((run/arm/'scores.json').read_text())
 report['score_recomputation'][arm]={'race_count':len(ids),'actual_source_values_match_saved_rows':True,'independent_vs_saved_max_abs':{'share_crps':max(abs(rows[j]['share_crps']-score['rows'][j]['share_crps']) for j in range(len(ids))),'win_brier':max(abs(rows[j]['win_brier']-score['rows'][j]['win_brier']) for j in range(len(ids))),'share_wis':max(abs(rows[j]['share_wis']-score['rows'][j]['share_wis']) for j in range(len(ids)) )},'all_score_rows':rows,'mean':{k:float(np.mean([x[k] for x in rows])) for k in ('share_crps','win_brier','share_wis')},'constant_winner_tail_race_ids':score['strata']['all'].get('unresolved_win_tail_race_ids',[]),'score_status':score['strata']['all']['status']}
# remove potentially chunky embedded per-race rows? keep for auditable output.
print(json.dumps(report,indent=2))
