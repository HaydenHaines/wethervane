"""Invented aggregate variance and empty-mask composition oracles."""
from pathlib import Path
import sys,json
import numpy as np
HERE=Path(__file__).resolve().parent;P=HERE.parent/'momentum-forecast-scoring-sol-v1';sys.path[:0]=[str(P),str(P/'snapshots')]
import score_math as s
scope='conditional_fixed_bank_Gibbs_integration'
def cycle(v,mcse):return dict(aggregates=dict(crps=dict(value=v,qualified=True,mcse=mcse,log_mcse=float(np.log(mcse)),uncertainty_scope=scope,status='qualified_conditional_numerical')))
a=s.equal_cycle_summary({2018:cycle(.1,.001),2022:cycle(.2,.002)},'crps',tier='A')
b=s.equal_cycle_summary({2018:cycle(.2,.003),2022:cycle(.3,.004)},'crps',tier='A')
d=s.paired_difference(dict(mask_race_ids=['2018:SYN','2022:SYN'],aggregates=dict(crps=a)),dict(mask_race_ids=['2018:SYN','2022:SYN'],aggregates=dict(crps=b)),'crps',independent_streams=True)
expected=np.sqrt(.001**2+.002**2+.003**2+.004**2)/2
assert d['mcse'] is not None and abs(d['mcse']-expected)<1e-16,d
empty=s.score_gaussian_forecast(np.empty((4,1600,0)),np.empty((4,1600,0)),[],[],[],[])
assert s.paired_difference(empty,empty,'crps',independent_streams=True)['value'] is None
missing=s.equal_cycle_summary({2018:cycle(.1,.001),2022:empty},'crps',tier='A')
assert missing['value'] is None and not missing['qualified']
analytic=dict(mask_race_ids=['SYN'],aggregates=dict(crps=dict(value=.2,qualified=True,mcse=0.,status='deterministic_analytic_no_MCMC',uncertainty_scope='deterministic_gaussian_analytic')))
stochastic=dict(mask_race_ids=['SYN'],aggregates=cycle(.1,.001)['aggregates'])
mixed=s.paired_difference(stochastic,analytic,'crps')
assert abs(mixed['mcse']-.001)<1e-16
zero=s.equal_cycle_summary({2018:analytic,2022:analytic},'crps',tier='A')
assert zero['mcse']==0 and zero['sampling_variance_zero']
nested=s.paired_difference(dict(mask_race_ids=['2018:SYN','2022:SYN'],aggregates=dict(crps=a)),dict(mask_race_ids=['2018:SYN','2022:SYN'],aggregates=dict(crps=zero)),'crps')
assert nested['mcse'] is not None and abs(nested['mcse']-a['mcse'])<1e-16,nested
out=dict(schema='IndependentSyntheticScoreCompositionEdgesSolV1',synthetic_only=True,primary_tier_mcse=float(d['mcse']),independent_expected_tier_mcse=float(expected),empty_2022_near_even_A_unavailable=True,deterministic_zero_variance_composes=True,nested_tier_stochastic_minus_PO5_preserves_MCSE=True,actual_target_data=False)
(HERE/'edge-checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
