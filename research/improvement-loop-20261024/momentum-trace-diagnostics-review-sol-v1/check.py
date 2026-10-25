"""Independent synthetic numerical diagnostic oracles; no historical inputs."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, norm

HERE=Path(__file__).resolve().parent
P=HERE.parent/'momentum-trace-diagnostics-sol-v1'
spec=importlib.util.spec_from_file_location('review_fixture',P/'test_trace_diagnostics.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
d=t.td
j=np.arange(1600)
x=np.array([np.sin(j/11+c)+.17*np.cos(j/53+c/5)+c*.003 for c in range(4)])
r=d.function_diagnostics(x[...,None],['synthetic'])['coordinates'][0]
# Direct partition calculations do not call the provider diagnostics.
vs=[np.var(x.mean(axis=1),ddof=1)/4]
for size in (50,100,200):
    means=np.array([chunk.mean() for chain in x for chunk in np.split(chain,1600//size)])
    vs.append(np.var(means,ddof=1)/len(means))
mcse=np.sqrt(max(vs));ess=np.var(x,ddof=1)/mcse**2
split=np.concatenate((x[:,:800],x[:,800:]),axis=0)
rh=[]
for z in (split,abs(split-np.median(split))):
    normal=norm.ppf((rankdata(z).reshape(z.shape)-.375)/(z.size+.25))
    W=np.var(normal,axis=1,ddof=1).mean()
    B=800*np.var(normal.mean(axis=1),ddof=1)
    rh.append(np.sqrt(((799/800)*W+B/800)/W))
assert abs(r['mcse']-mcse)<1e-15 and abs(r['mean_ess_estimate']-ess)<1e-11
assert abs(r['rank_folded_split_rhat']-max(rh))<1e-15
base=(-2+.1*x)[...,None]
a=d.log_tail_diagnostics(base,['D'])['coordinates'][0]
b=d.log_tail_diagnostics(base-10000,['D'])['coordinates'][0]
assert abs(a['relative_mcse']-b['relative_mcse'])<1e-12
assert abs(a['log_probability']-b['log_probability']-10000)<1e-11
assert b['absolute_mcse_underflows'] and b['absolute_mcse'] is None
assert d.raw_indicator_diagnostics(np.zeros((4,1600,1)),['R'])['coordinates'][0]['status']=='unresolved_constant'
# Unequal per-chain retained counts use count-weighted pooling, not chain averaging.
w=np.array([[.1,.2,.3,.4],[.4,.3,.2,.1],[.2,.3,.1,.4],[.3,.1,.4,.2]])
counts=np.array([[20,30],[10,40],[25,35],[30,40]])
sums=counts[...,None]*w[:,None,:]
q=d.bank_diagnostics(sums,counts,[0,0,1,1]);mass=(w*counts.sum(axis=1)[:,None]).sum(axis=0)/counts.sum()
assert np.allclose(q['row_mass'],mass,atol=1e-16,rtol=0)
assert abs(q['effective_bank_rows']-1/(mass@mass))<1e-14
assert abs(q['entropy']+np.sum(mass*np.log(mass)))<1e-14
out=dict(schema='IndependentSyntheticTraceDiagnosticOraclesSolV1',synthetic_only=True,mcse_error=abs(r['mcse']-mcse),mean_ess_error=abs(r['mean_ess_estimate']-ess),rhat_error=abs(r['rank_folded_split_rhat']-max(rh)),tail_scale_invariance_absolute_log_shift=10000,constant_raw_tail_unresolved=True,bank_pooled_mass_verified=True)
(HERE/'checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
# Independent every-event observer-prefix reconstruction, including pending row mass.
prior,_,_,records,_=t.synthetic()
sink=t.collector(prior)
completed=[0]*4
phase_weights={}
accepted_weights={c:[] for c in range(4)}
occupancy=np.zeros((4,12),dtype=int)
finals=0
for event in [None,*records]:
    if event is not None:
        c=event['chain'];phase=event['phase'];sink(event)
        if phase=='row_weights':phase_weights[c]=event['normalized_row_weights']
        elif phase=='completed_sweep':
            completed[c]+=1;accepted_weights[c].append(phase_weights.pop(c));occupancy[c,event['sampled_row_index']]+=1
        elif phase=='chain_final_rng':finals+=1
    part=sink.export_partial();meta=part['metadata'];arr=part['arrays']
    assert meta['observer_phase_prefix_only'] and not meta['raw_predictive_draw_completeness_claimed'] and not meta['restartable']
    assert list(meta['completed_draws_per_chain'])==completed
    assert len(meta['final_rng_states_emitted'])==finals
    for c,n in enumerate(completed):
        prefix=f'chain{c}__'
        assert arr[prefix+'observed_theta'].shape[1]==n
        assert np.array_equal(arr[prefix+'bank_sampled_occupancy'][0],occupancy[c])
        for b in range((n+3)//4):
            oracle=np.sum(accepted_weights[c][b*4:min(n,b*4+4)],axis=0)
            assert np.allclose(arr[prefix+'bank_batch_weight_sums'][0,b],oracle,atol=2e-15,rtol=0)
out['every_event_observer_prefix_boundaries_verified']=len(records)+1
(HERE/'checks.json').write_text(json.dumps(out,indent=2)+'\n')
