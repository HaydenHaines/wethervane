"""Importance integration for an actual-coordinate six-poll mixture subset.

No fitting, outcome labels, provider registration, or model publication.
"""
import hashlib, importlib.util, json, platform, sys
from pathlib import Path
import numpy as np
import scipy
from scipy.special import expit, logit, logsumexp, ndtri
from scipy.stats import multivariate_normal, qmc

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCE = BASE / 'm01-count-measurement-prototype-sol-v1/prototype.py'
CONSUMER = BASE / 'calibration-consumer-v1'

def pin(path):
    data = Path(path).read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

def main():
    spec = importlib.util.spec_from_file_location('m01_normalization_prototype', SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    case = module.load_frozen_current_case(CONSUMER)
    full = case['mixture']
    indexes = [0, len(full.weights)//3, 2*len(full.weights)//3, len(full.weights)-1]
    if len(set(indexes)) != 4: raise ValueError('Four distinct calibration components required')
    # These fixed masses are an invented sensitivity, not the original 4,000-row bank.
    weights = np.array([.1,.2,.3,.4])
    cov = full.covariances[indexes]
    model = module.SharedVectorMixture(cov, weights)
    mean = np.full(model.dimension, .15)
    if model.dimension != 6: raise ValueError('Expected the complete six-poll vector')
    proposal_cov = np.einsum('s,sij->ij', weights, cov) * 1.5
    lower = np.linalg.cholesky(proposal_cov)
    precision = np.linalg.inv(proposal_cov)
    proposal_constant = -.5*(6*np.log(2*np.pi)+np.linalg.slogdet(proposal_cov)[1])
    means=[]; missing_jac=[]; ref_errors=[]; range_errors=[]
    for seed in (31,47,59,71,83,97,101,127):
        u = qmc.Sobol(d=6, scramble=True, seed=seed).random_base2(m=14)
        z = ndtri(u); logits = mean + z @ lower.T
        shares = expit(logits)
        if not ((shares > 0).all() and (shares < 1).all()): raise ValueError('share boundary reached')
        back = logit(shares)
        range_errors.append(float(np.max(np.abs(logits-back))))
        residual = back - mean
        component_quad = np.einsum('ni,sij,nj->ns',residual,model.precisions,residual,optimize=True)
        log_target = logsumexp(model.log_weights + model.log_constants - .5*component_quad, axis=1)
        log_proposal = proposal_constant - .5*np.einsum('ni,ij,nj->n',residual,precision,residual,optimize=True)
        log_jac = -np.log(shares*(1-shares)).sum(axis=1)
        # Both are densities on the share cube, each transformed from its own logit density.
        ratio = np.exp((log_target + log_jac) - (log_proposal + log_jac))
        means.append(float(ratio.mean()))
        missing_jac.append(float(np.exp(log_target-(log_proposal+log_jac)).mean()))
        # Cross-check actual scalar implementation and independent scipy components.
        for i in (0,11,137):
            got=model.evaluate(back[i],mean)[0]
            expected=logsumexp([np.log(w)+multivariate_normal.logpdf(back[i],mean,c) for w,c in zip(weights,cov,strict=True)])
            ref_errors.append(abs(float(got)-float(expected)))
    result={'schema':'M01SixPollShareImportanceNormalizationV1',
        'sourcePins':{'prototype':pin(SOURCE),'consumer':pin(CONSUMER/'consumer.py'), 'manifest':pin(CONSUMER/'input-manifest.json'), 'script':pin(HERE/'check.py')},
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'dimension':model.dimension, 'components':indexes, 'subset_weights':weights.tolist(), 'proposal':'one logistic-normal with covariance 1.5 times the weighted mixture covariance; not the target mixture',
        'points_per_scramble':2**14,'scramble_seeds':[31,47,59,71,83,97,101,127],
        'integral_estimates':means,'mean_estimate':float(np.mean(means)), 'between_scramble_sd':float(np.std(means,ddof=1)),
        'missing_jacobian_estimates':missing_jac,'scalar_scipy_max_abs_error':max(ref_errors),'logit_roundtrip_max_abs_error':max(range_errors),
        'scope':'Numerical integration of an invented four-component reweighting of actual six-poll covariance components, at a supplied constant mean. Proposal differs from target. Tests the reconstructed share-measure mixture and sampled scalar implementation; it is not the full 4000-row bank, a fitted posterior, native factor, source admission, or empirical forecast validation.'}
    if abs(result['mean_estimate']-1) > .02: raise AssertionError('importance estimate misses 1 by more than two percent')
    if not (max(missing_jac) < .01 and max(ref_errors) < 1e-9 and max(range_errors) < 1e-10): raise AssertionError('negative control, scalar parity or transform failed')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
