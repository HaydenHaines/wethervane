"""Independent supplied synthetic reference algebra, never real archive reads."""
from pathlib import Path
import sys
import json
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
P=HERE.parent/'momentum-forecast-execution-sol-v1'
sys.path.insert(0,str(P))
import references as ref
import runner
idx=np.array([0,2,0]);R=np.array([[.13,.02,.03],[.02,.19,.01],[.03,.01,.17]])
y=np.array([.2,-.1,.3]);n=4
m,C,receipt=ref.analytic_po5(SimpleNamespace(race_indices=idx,covariance=R,values=y),n)
H=np.eye(n)[idx];Q=np.eye(n)/6.25+H.T@np.linalg.inv(R)@H
cov=np.linalg.inv(Q);mean=cov@H.T@np.linalg.inv(R)@y
assert np.max(abs(m-mean))<1e-14 and np.max(abs(C-cov))<1e-14
assert m[1]==m[3]==0 and C[1,1]==C[3,3]==6.25
seeds={str((yr,arm,k)):runner.job_settings(yr,arm,k) for yr in (2018,2020,2022,2024) for arm in ('FS5','FS6','LP5') for k in (0,1)}
for settings in seeds.values():
 expected=int(np.random.SeedSequence([2026102501,settings['cycle'],{'FS5':1,'FS6':2,'LP5':3}[settings['arm']],settings['attempt']]).generate_state(1,dtype=np.uint64)[0])
 assert settings['seed']==expected
 assert settings['draws']==(4000 if settings['attempt']==0 else 8000)
assert len({v['seed'] for v in seeds.values()})==24
out=dict(schema='IndependentForecastExecutionSyntheticReferenceCheckV1',synthetic_only=True,po5_mean_error=float(np.max(abs(m-mean))),po5_covariance_error=float(np.max(abs(C-cov))),unpolled_independent_prior_preserved=True,all_24_seed_budget_identities_verified=True,actual_target_chains=False,archive_numeric_values=False,target_labels=False)
(HERE/'checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
