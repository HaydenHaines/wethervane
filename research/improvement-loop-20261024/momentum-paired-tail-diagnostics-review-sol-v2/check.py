"""Independent synthetic complementary-tail oracles; no target data."""
from decimal import Decimal,localcontext
from pathlib import Path
import importlib.util,json,sys
import numpy as np
from scipy.special import log_ndtr
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE/'momentum-trace-diagnostics-sol-v1/snapshots'))
sys.path.insert(0,str(BASE/'momentum-trace-diagnostics-sol-v1'))
sys.path.insert(0,str(BASE/'momentum-paired-tail-diagnostics-sol-v2'))
import paired_tail as p
j=np.arange(1600)
z=np.array([9+.5*(1+np.sin(j/17+c)) for c in range(4)])
lw,ll=log_ndtr(z)[...,None],log_ndtr(-z)[...,None]
r=p.paired_tail_diagnostics(lw,ll,['synthetic'])
a,b=(r[s]['coordinates'][0] for s in ('win','lose'))
small=np.exp(ll[:,:,0])
vs=[np.var(small.mean(axis=1),ddof=1)/4]
for size in (50,100,200):
 batch=np.array([part.mean() for chain in small for part in np.split(chain,1600//size)])
 vs.append(np.var(batch,ddof=1)/len(batch))
mcse=np.sqrt(max(vs))
assert abs(a['log_absolute_mcse']-np.log(mcse))<1e-13
assert a['log_absolute_mcse']==b['log_absolute_mcse']
assert a['status']=='measured' and a['original_one_sided_status']=='unresolved_constant'
assert np.all(np.exp(lw)==1)
# Decimal complement of represented small means, independent of floating1-p.
with localcontext() as ctx:
 ctx.prec=90
 pooled=sum(Decimal(float(x)) for x in small.ravel())/Decimal(small.size)
 loglarge=float((Decimal(1)-pooled).ln())
 chainlogs=[float((Decimal(1)-sum(Decimal(float(x)) for x in row)/Decimal(len(row))).ln()) for row in small]
 assert abs(a['log_probability']-loglarge)<1e-32
 assert max(abs(np.asarray(a['per_chain_log_difference'])-(np.array(chainlogs)-loglarge)))<1e-32
swapped=p.paired_tail_diagnostics(ll,lw,['synthetic'])
assert swapped['lose']['coordinates'][0]['log_absolute_mcse']==a['log_absolute_mcse']
c=p.probability_coordinates(lw,ll,['synthetic'])
assert np.array_equal(c['win_deviation_sign'],-c['lose_deviation_sign'])
assert not c['centered_scaled_small'].flags.writeable
out=dict(schema='IndependentPairedTailSyntheticCheckV2',synthetic_only=True,large_probability_rounds_to_one=True,paired_large_status=a['status'],preserved_original_one_sided_status=a['original_one_sided_status'],direct_log_absolute_mcse_error=abs(a['log_absolute_mcse']-np.log(mcse)),decimal_complement_log_probability_error=abs(a['log_probability']-loglarge),decimal_chain_log_difference_error=float(max(abs(np.asarray(a['per_chain_log_difference'])-(np.array(chainlogs)-loglarge)))),side_swap_and_negative_complement_verified=True)
(HERE/'checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

# Both contradiction orientations reject before any normalization adjustment.
for w,l in ((lw,ll+.1),(ll+.1,lw)):
    try:p.probability_coordinates(w,l,['synthetic'])
    except ValueError as e:assert 'Recoverably inconsistent' in str(e)
    else:raise AssertionError('Recoverable contradiction accepted')
zgrid=np.linspace(-40,40,100001).reshape(1,-1,1)
grid=p.probability_coordinates(log_ndtr(zgrid),log_ndtr(-zgrid),['grid'])
assert int(grid['relative_cross_validation_recoverable_count'][0])==93799
assert int(grid['relative_cross_validation_unavailable_count'][0])==6202
assert float(grid['recoverable_max_log_error'][0])<1e-12
out['v1_contradiction_rejected_both_orientations']=True
out['native_gaussian_grid_recoverable_count']=93799
out['native_gaussian_grid_unavailable_count']=6202
out['grid_max_recoverable_log_error']=float(grid['recoverable_max_log_error'][0])
(HERE/'checks.json').write_text(json.dumps(out,indent=2)+'\n')
