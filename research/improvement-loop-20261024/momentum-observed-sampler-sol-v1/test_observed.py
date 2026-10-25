"""Only deterministic fixtures and explicitly authorized small synthetic Gibbs chains."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import MappingProxyType
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'snapshots'))
sys.path.insert(0, str(HERE))
import numpy as np
import pytest
from scipy.special import logsumexp
from src.forecasting import polling_basis_v1 as original
from src.forecasting.polling_v1 import GaussianPollLikelihood, compress_likelihood
import observed_sampler as observed


def digest(value):
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, dict):
        return {k: digest(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(digest(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def assert_equal(a, b):
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()
    elif isinstance(a, dict) or isinstance(a, MappingProxyType):
        assert set(a) == set(b)
        for k in a:
            assert_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert type(a) is type(b) and a == b


class TrackedGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.calls = []

    @property
    def bit_generator(self):
        return self.generator.bit_generator

    def __getattr__(self, name):
        method = getattr(self.generator, name)
        def tracked(*args, **kwargs):
            result = method(*args, **kwargs)
            self.calls.append((name, digest(args), digest(kwargs), digest(result)))
            return result
        return tracked


def tracked_run(module, prior, polls, **kwargs):
    native_factory = np.random.default_rng
    generators = []
    def factory(seed):
        wrapped = TrackedGenerator(native_factory(seed))
        generators.append(wrapped)
        return wrapped
    with patch.object(np.random, 'default_rng', factory):
        result = module.sample_poll_posterior(prior, polls, **kwargs)
    return result, [g.calls for g in generators], [copy.deepcopy(g.bit_generator.state) for g in generators]


def fixture(module, path='missing', *, rows=12, q=3, n=4):
    # All numeric data are algebraically constructed synthetic values, no files.
    beta = .12 * np.sin(np.arange(rows * q).reshape(rows, q) / 3)
    design = np.cos(np.arange(n * q).reshape(n, q) / 5)
    kwargs = dict(offset=np.linspace(-.3,.3,n), design=design, beta=beta,
                  tau_cycle=.13 + .04*(np.arange(rows)%3),
                  tau_race=.25 + .05*(np.arange(rows)%4),
                  race_ids=tuple(f'SYNTHETIC:{r}' for r in range(n)),
                  source_identity='synthetic-only-no-target-input',
                  basis_names=tuple(f'b{j}' for j in range(q)),
                  posterior_chain=np.arange(rows)//(rows//4),
                  posterior_draw=np.arange(rows)%(rows//4),
                  explained_national_mean=np.linspace(-.1,.1,rows),
                  metadata={'nested': {'units': 'synthetic logits', 'items': [1,2]},
                            'implementation_id': 'synthetic', 'target_id': 'no-election'})
    prior = module.EmpiricalBasisFundamentals(**kwargs)
    if path == 'none':
        index = np.array([],dtype=int)
    elif path == 'full':
        index = np.r_[np.arange(n),0,n-1]
    else:
        index = np.r_[np.arange(n-1),0,n-2]
    m = len(index)
    covariance = .04*np.eye(m) + .006*np.ones((m,m))
    polls = GaussianPollLikelihood(index, .15*np.sin(np.arange(m)), covariance,
                                   tuple(f'SYNTHETIC:poll:{j}' for j in range(m)))
    return prior, polls


@pytest.mark.parametrize('path', ['none', 'missing', 'full'])
@pytest.mark.parametrize('warmup', [0, 7])
@pytest.mark.parametrize('seed', [0, 78341])
def test_original_off_on_exact_real_chain_parity(path, warmup, seed):
    p0, polls = fixture(original, path)
    p1, _ = fixture(observed, path)
    kwargs = dict(chains=4,warmup=warmup,draws=20,seed=seed)
    base, base_calls, base_final = tracked_run(original,p0,polls,**kwargs)
    off, off_calls, off_final = tracked_run(observed,p1,polls,**kwargs)
    records=[]
    collector=observed.RetainedRowWeightAccumulator(batch_length=6,downstream=records.append)
    on,on_calls,on_final=tracked_run(observed,p1,polls,observer=collector,**kwargs)
    assert_equal(base,off)
    assert_equal(base,on)
    assert_equal(base_calls,off_calls)
    assert_equal(base_calls,on_calls)
    assert_equal(base_final,off_final)
    assert_equal(base_final,on_final)
    final=[r for r in records if r['phase']=='chain_final_rng']
    assert len(final)==4
    for c,r in enumerate(final):
        assert r['chain']==c and r['retained_draws']==20
        assert_equal(base_final[c], r['state'])
    exported=collector.export()
    assert all(sum(v.values())==20 for v in exported['occupancy'].values())
    if path=='none':
        assert len(records)==8
        assert not exported['counts'] and not exported['batch_weight_sums']
        for w in exported['uniform_prior_weights'].values():
            np.testing.assert_array_equal(w,np.full(12,1/12))
    else:
        assert len(records)==4*(20*3+1)
        assert dict(exported['counts'])=={c:20 for c in range(4)}
        assert len(exported['batch_weight_sums'])==16 # 6+6+6+2 per chain
        for (c,b),s in exported['batch_weight_sums'].items():
            count=6 if b<3 else 2
            assert abs(s.sum()-count)<2e-14
            expected=sum((r['normalized_row_weights'] for r in records
                          if r['phase']=='row_weights' and r['chain']==c
                          and r['retained_draw']//6==b),np.zeros(12))
            np.testing.assert_array_equal(s,expected)


def test_supplied_dense_gaussian_and_row_law_phase_identity():
    prior,polls=fixture(observed,'missing')
    records=[]
    result=observed.sample_poll_posterior(prior,polls,chains=4,warmup=7,draws=20,seed=90210,observer=records.append)
    obs=sorted(set(polls.race_indices.tolist()))
    H=np.zeros((len(polls.values),len(obs)))
    for j,race in enumerate(polls.race_indices):
        H[j,obs.index(int(race))]=1
    P=H.T @ np.linalg.solve(polls.covariance,H)
    h=H.T @ np.linalg.solve(polls.covariance,polls.values)
    changed=0
    for chain in range(4):
        prev=None
        for draw in range(20):
            group={r['phase']:r for r in records if r['chain']==chain and r.get('retained_draw')==draw}
            a,w,b=(group[k] for k in ['gaussian_conditional','row_weights','completed_sweep'])
            k=a['conditioning_row_index']; lam=a['conditioning_lambda']
            if prev is not None:
                assert k==prev['sampled_row_index']
                np.testing.assert_array_equal(lam,prev['lambda_after_gamma'])
            assert w['conditioning_row_index']==k
            np.testing.assert_array_equal(w['conditioning_lambda'],lam)
            np.testing.assert_array_equal(w['theta'],result['logits'][chain,draw,obs])
            assert w['cycle']==b['cycle']==result['shared_cycle'][chain,draw]
            assert b['sampled_row_index']==result['posterior_index'][chain,draw]
            changed+=k!=b['sampled_row_index']
            np.testing.assert_array_equal(result['posterior_chain'][chain,draw],prior.posterior_chain[b['sampled_row_index']])
            np.testing.assert_array_equal(result['posterior_draw'][chain,draw],prior.posterior_draw[b['sampled_row_index']])
            loc=prior.offset[obs]+prior.design[obs]@prior.beta[k]
            d=lam/prior.tau_race[k]**2
            J=np.block([[P+np.diag(d),-d[:,None]],[-d[None,:],np.array([[d.sum()+1/prior.tau_cycle[k]**2]])]])
            rhs=np.r_[h+d*loc,-d@loc]
            np.testing.assert_allclose(a['mean_theta_cycle'],np.linalg.solve(J,rhs),atol=2e-15,rtol=2e-14)
            np.testing.assert_allclose(a['marginal_variances'],np.diag(np.linalg.inv(J)),atol=2e-15,rtol=2e-14)
            # Independent dense residual oracle includes every row-dependent normalizer.
            residual=w['theta'][None,:]-prior.offset[obs]-prior.beta@prior.design[obs].T-w['cycle']
            logw=-np.log(prior.tau_cycle)-len(obs)*np.log(prior.tau_race)-.5*(w['cycle']/prior.tau_cycle)**2-.5*np.sum(lam*residual**2,axis=1)/prior.tau_race**2
            np.testing.assert_allclose(w['normalized_row_weights'],np.exp(logw-logsumexp(logw)),atol=1e-15,rtol=2e-14)
            missing=b['missing_race_indices']
            np.testing.assert_array_equal(missing,[3])
            np.testing.assert_allclose(b['missing_location'],prior.offset[missing]+prior.design[missing]@prior.beta[b['sampled_row_index']]+b['cycle'],atol=0,rtol=0)
            assert b['missing_scale']==prior.tau_race[b['sampled_row_index']]
            prev=b
    assert changed>20 # Demonstrate pre/post identity distinction is actually exercised.


def test_callback_copies_are_immutable_even_if_write_flag_is_requested():
    prior,polls=fixture(observed,'missing')
    records=[]
    attempts=[]
    def adversarial(record):
        records.append(record)
        with pytest.raises(TypeError):
            record['phase']='corrupt'
        def visit(value):
            if isinstance(value,np.ndarray):
                assert not value.flags.writeable
                with pytest.raises(ValueError):
                    value.setflags(write=True)
                if value.size:
                    with pytest.raises(ValueError):
                        value.flat[0]=999
                detached=value.copy()
                if detached.size:
                    detached.flat[0]=999
                attempts.append(value.shape)
            elif isinstance(value,MappingProxyType):
                with pytest.raises(TypeError):
                    value['new_key']='corrupt'
                for item in value.values():
                    visit(item)
        visit(record)
    baseline,calls,states=tracked_run(observed,prior,polls,chains=4,warmup=3,draws=10,seed=99)
    result,calls2,states2=tracked_run(observed,prior,polls,chains=4,warmup=3,draws=10,seed=99,observer=adversarial)
    assert attempts
    assert_equal(baseline,result)
    assert_equal(calls,calls2)
    assert_equal(states,states2)
    # Data remain detached after subsequent iterations and external copies mutate.
    assert all(r['phase'] in {'gaussian_conditional','row_weights','completed_sweep','chain_final_rng'} for r in records)


def test_default_signature_and_noncallable_rejection():
    import inspect
    a=inspect.signature(original.sample_poll_posterior)
    b=inspect.signature(observed.sample_poll_posterior)
    assert list(b.parameters)==list(a.parameters)+['observer']
    for key in a.parameters:
        assert a.parameters[key]==b.parameters[key]
    assert b.parameters['observer'].default is None
    prior,polls=fixture(observed)
    with pytest.raises(ValueError,match='Callable observer'):
        observed.sample_poll_posterior(prior,polls,observer=object())


def test_observer_errors_are_explicit():
    prior,polls=fixture(observed)
    def broken(record):
        raise RuntimeError('synthetic callback failed')
    with pytest.raises(RuntimeError,match='synthetic callback failed'):
        observed.sample_poll_posterior(prior,polls,chains=4,warmup=0,draws=10,observer=broken)


def test_synthetic_large_shape_smoke_exact_parity(tmp_path):
    # 16000 rows / q6 / 30 observed + 2 missing. Only 4*(2+10) sweeps.
    p0,polls=fixture(original,rows=16000,q=6,n=32)
    # fixture 'missing' normally observes n-1; rebuild with exactly30 observed.
    ix=np.r_[np.arange(30),0,29]
    m=len(ix)
    polls=GaussianPollLikelihood(ix,.1*np.sin(np.arange(m)),.04*np.eye(m)+.002*np.ones((m,m)),tuple(f'SYNTHETIC:large:{j}' for j in range(m)))
    p1,_=fixture(observed,rows=16000,q=6,n=32)
    collector=observed.RetainedRowWeightAccumulator(batch_length=5)
    start=time.perf_counter()
    base,bc,bs=tracked_run(original,p0,polls,chains=4,warmup=2,draws=10,seed=217)
    middle=time.perf_counter()
    result,rc,rs=tracked_run(observed,p1,polls,chains=4,warmup=2,draws=10,seed=217,observer=collector)
    end=time.perf_counter()
    assert_equal(base,result); assert_equal(bc,rc); assert_equal(bs,rs)
    assert len(collector.export()['batch_weight_sums'])==8
    assert end-start<120 # Synthetic bounded smoke only, not scientific adequacy.
    (tmp_path/'synthetic-large-smoke.json').write_text(json.dumps(dict(
        synthetic_only=True,rows=16000,basis=6,observed_races=30,missing_races=2,
        chains=4,warmup=2,draws=10,original_seconds=middle-start,observed_seconds=end-middle,
        exact_original_outputs_and_rng_calls_and_final_states=True),indent=2)+'\n')


def test_precision_records_match_actual_gamma_rng_results_and_conditionals():
    prior,polls=fixture(observed,'missing')
    records=[]
    result,calls,_=tracked_run(observed,prior,polls,chains=4,warmup=3,draws=10,seed=441,observer=records.append)
    obs=result['observed_race_indices']
    for chain in range(4):
        gamma_calls=[call for call in calls[chain] if call[0]=='gamma']
        assert len(gamma_calls)==1+3+10 # initialization, all warmup and retained steps
        for draw in range(10):
            r=next(r for r in records if r['phase']=='completed_sweep' and r['chain']==chain and r['retained_draw']==draw)
            name,args,kwargs,output=gamma_calls[1+3+draw]
            assert output==digest(r['lambda_after_gamma'])
            k=r['sampled_row_index']
            location=prior.offset[obs]+prior.design[obs]@prior.beta[k]
            residual=(result['logits'][chain,draw,obs]-location-r['cycle'])/prior.tau_race[k]
            assert args==digest(((prior.race_df+1)/2,2/(prior.race_df+residual**2)))
            assert kwargs=={}
