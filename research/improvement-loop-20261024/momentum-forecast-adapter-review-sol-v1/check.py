"""Independent read-only full bank/measurement reconstruction; no predictive values."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
P = HERE.parent/'momentum-forecast-adapter-sol-v1'
sys.path.insert(0,str(P))
import adapter as a
import numpy as np

# Deliberately prohibit execution of sampling, compilation with outcome labels,
# optimizer and archive predictive-value readers.
def blocked(frame,event,arg):
    if event == 'call' and frame.f_code.co_name in {'sample_poll_posterior','compile_fold','minimize','predict'}:
        raise AssertionError('Forbidden scientific operation: '+frame.f_code.co_name)
sys.setprofile(blocked)
rows=[]
for year in (2018,2020,2022,2024):
    x=a.load_adapter(year)
    prior,_=x.build_prior()
    for chain,source in enumerate(x.source_pins['raw_banks']):
        handle=a.az.from_netcdf(source['path'])
        try:
            for name in a.POSTERIOR_VARIABLES:
                assert np.array_equal(x.original_parameter_rows[name][chain*4000:(chain+1)*4000],handle.posterior[name].values[0])
        finally:handle.close()
    variants=[]
    for name,m in x.measurements.items():
        c=m.coordinates;n=len(c.observation_ids);d=c.days_to_election
        cy=np.asarray(c.cycles);ra=np.asarray(c.race_ids);po=np.asarray(c.pollsters)
        C=(cy[:,None]==cy);R=C&(ra[:,None]==ra)
        kernels=[np.eye(n),C,R,C&(po[:,None]==po),R*np.minimum.outer(d,d)/30,R*np.outer(d/30,d/30)]
        scales=np.array([m.scales[key] for key in a.MODEL.COMPONENTS])
        oracle=sum(s*s*K for s,K in zip(scales,kernels))
        delta=float(np.max(abs(oracle-m.likelihood_kwargs['covariance'])))
        assert delta<1e-15
        assert np.array_equal(m.likelihood_kwargs['race_indices'],np.array([prior.race_ids.index(r) for r in c.race_ids]))
        assert np.allclose(m.likelihood_kwargs['values'],np.log(m.shares)-np.log1p(-m.shares),atol=1e-15,rtol=0)
        variants.append(dict(arm=name,rows=n,covariance_error=delta))
    assert np.array_equal(x.uniform_prior_weights,np.full(16000,1/16000))
    assert np.array_equal(prior.posterior_chain,np.repeat(np.arange(4),4000))
    assert np.array_equal(prior.posterior_draw,np.tile(np.arange(4000),4))
    rows.append(dict(cycle=year,joint_rows=16000,races=len(prior.race_ids),variants=variants,source_identity=x.source_identity))
sys.setprofile(None)
(HERE/'checks.json').write_text(json.dumps(dict(schema='ForecastAdapterIndependentReconstructionV1',cycles=rows,archive_predictive_values_read=False,target_numeric_labels_read=False,sampler_or_optimizer_called=False),indent=2)+'\n')
print(json.dumps(rows,indent=2))
