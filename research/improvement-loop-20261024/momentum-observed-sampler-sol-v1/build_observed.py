"""Build a private observational copy; fail closed on source byte or edit-anchor drift."""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent
original = HERE / 'snapshots/src/forecasting/polling_basis_v1.py'
assert hashlib.sha256(original.read_bytes()).hexdigest() == 'd0722a26b02f0e9e0d54bdc73036b1b08d7720a27dc6c36e7ff8ead548747142'
s = original.read_text()

def replace_once(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)

replace_once('from scipy.special import expit', 'from scipy.special import expit, logsumexp')
replace_once('def sample_poll_posterior(prior, polls, *, chains=4, warmup=1000, draws=4000, seed=202610201):', 'def sample_poll_posterior(prior, polls, *, chains=4, warmup=1000, draws=4000, seed=202610201, observer=None):')
replace_once('    observed, precision, linear = compress_likelihood(prior, polls)\n', '    require(observer is None or callable(observer), "Callable observer required")\n    observed, precision, linear = compress_likelihood(prior, polls)\n')
replace_once('            cycles[chain], indices[chain] = c, rows\n            continue', '''            cycles[chain], indices[chain] = c, rows
            if observer is not None:
                _emit(observer, phase="direct_prior_chain", chain=chain,
                      retained_draws=draws, sampled_row_indices=rows,
                      shared_cycles=c, uniform_row_weights=np.full(k, 1 / k))
                _emit(observer, phase="chain_final_rng", chain=chain,
                      retained_draws=draws, state=rng.bit_generator.state)
            continue''')
replace_once('            state = mean + solve_triangular(root.T, rng.normal(size=len(observed) + 1), lower=False)', '''            if observer is not None and step >= warmup:
                inverse_root = solve_triangular(root, np.eye(len(observed) + 1), lower=True)
                _emit(observer, phase="gaussian_conditional", chain=chain,
                      retained_draw=step - warmup, conditioning_row_index=row,
                      conditioning_lambda=lam, mean_theta_cycle=mean,
                      marginal_variances=np.sum(inverse_root * inverse_root, axis=0))
            state = mean + solve_triangular(root.T, rng.normal(size=len(observed) + 1), lower=False)''')
replace_once('            row = _categorical(conditional_row_log_weights(prior, observed, theta, cycle, lam), rng)', '''            if observer is not None and step >= warmup:
                observation_log_weights = conditional_row_log_weights(prior, observed, theta, cycle, lam)
                _emit(observer, phase="row_weights", chain=chain,
                      retained_draw=step - warmup, conditioning_row_index=row,
                      conditioning_lambda=lam, theta=theta, cycle=cycle,
                      normalized_row_weights=np.exp(observation_log_weights - logsumexp(observation_log_weights)))
            row = _categorical(conditional_row_log_weights(prior, observed, theta, cycle, lam), rng)''')
replace_once('            if step >= warmup:\n                j = step - warmup', '''            if observer is not None and step >= warmup:
                _emit(observer, phase="completed_sweep", chain=chain,
                      retained_draw=step - warmup, sampled_row_index=row,
                      lambda_after_gamma=lam, cycle=cycle,
                      missing_location=prior.offset[missing] + prior.design[missing] @ prior.beta[row] + cycle,
                      missing_scale=prior.tau_race[row], observed_race_indices=observed,
                      missing_race_indices=missing)
            if step >= warmup:
                j = step - warmup''')
replace_once('                cycles[chain, j], indices[chain, j] = cycle, row\n    return dict(', '''                cycles[chain, j], indices[chain, j] = cycle, row
        if observer is not None:
            _emit(observer, phase="chain_final_rng", chain=chain,
                  retained_draws=draws, state=rng.bit_generator.state)
    return dict(''')
helpers = '''

def _observed_copy(value):
    """Deep immutable callback data with no aliases into transition buffers.

    Arrays are backed by bytes, so setflags(write=True) is rejected as well as
    ordinary writes. Even a consumer's own mutable copy cannot reach a sampler.
    """
    if isinstance(value, np.ndarray):
        return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _observed_copy(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_observed_copy(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    require(isinstance(value, (str, int, float, bool, type(None))), "Unsupported observation value")
    return value


def _emit(observer, **values):
    observer(_observed_copy(values))


class RetainedRowWeightAccumulator:
    """Callback retaining all-bank mass and count summaries, never RNG state.

    One fixed contiguous block per chain/batch; no pruning or row renaming.
    Direct-prior branches record their uniform law separately from row updates.
    Exported observations are deep immutable copies. Optional downstream sees
    the same immutable event and has no generator or transition-array alias.
    """
    def __init__(self, *, batch_length=100, downstream=None):
        require(type(batch_length) is int and batch_length > 0, "Positive batch length required")
        require(downstream is None or callable(downstream), "Callable downstream required")
        self.batch_length = batch_length
        self.downstream = downstream
        self._batches = {}
        self._counts = {}
        self._occupancy = {}
        self._uniform_prior = {}
        self._rng_states = {}

    def __call__(self, record):
        phase, chain = record["phase"], record["chain"]
        if phase == "row_weights":
            draw = record["retained_draw"]
            key = (chain, draw // self.batch_length)
            weights = record["normalized_row_weights"]
            require(draw == self._counts.get(chain, 0), "Ordered contiguous retained row-weight events required")
            if key not in self._batches:
                self._batches[key] = np.zeros_like(weights)
            self._batches[key] += weights
            self._counts[chain] = draw + 1
        elif phase == "completed_sweep":
            row = record["sampled_row_index"]
            counts = self._occupancy.setdefault(chain, {})
            counts[row] = counts.get(row, 0) + 1
        elif phase == "direct_prior_chain":
            self._uniform_prior[chain] = np.array(record["uniform_row_weights"], copy=True)
            rows, counts = np.unique(record["sampled_row_indices"], return_counts=True)
            self._occupancy[chain] = dict(zip(rows.tolist(), counts.tolist()))
        elif phase == "chain_final_rng":
            self._rng_states[chain] = record["state"]
        if self.downstream is not None:
            self.downstream(record)

    def export(self):
        return _observed_copy(dict(batch_length=self.batch_length, counts=self._counts,
                                   batch_weight_sums=self._batches, occupancy=self._occupancy,
                                   uniform_prior_weights=self._uniform_prior,
                                   final_rng_states=self._rng_states))
'''
replace_once('\ndef sample_poll_posterior(', helpers + '\n\ndef sample_poll_posterior(')
(HERE / 'observed_sampler.py').write_text(s)
print('Built private observed_sampler.py; original transition source retained verbatim between observation blocks.')
