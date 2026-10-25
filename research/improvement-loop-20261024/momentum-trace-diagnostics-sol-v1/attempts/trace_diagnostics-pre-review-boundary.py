"""Strict observer collection and supplied numerical diagnostics, no label/score I/O.

All substantive diagnostic gates are explicit reviewable configuration. No global
forecast qualification, retry, model selection, marginal scoring or fitting.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.special import logsumexp
from observed_sampler import RetainedRowWeightAccumulator, _observed_copy
from src.forecasting.response_v5_reuse import correlated_mcse


ROOT_DESIGN_SHA256='45707d4a933ded2f7d4bffcbaab0fb6fb9a4ab7b1d62ad4494a27422b7d75678'
REQUIRED_SOURCE_PINS=MappingProxyType(dict(observed_sampler_manifest='5e8608c4d1de978e8119a8cd1b4dded46ffe6253c4ff973f53e20d2bf98402c2',
    marginal_math_manifest='2bfa468bca013c7bfa7d6e80395ba7534947026ff35209fd2d51fed1b40f28c8',
    forecast_design=ROOT_DESIGN_SHA256,correlated_mcse_source='d4330647dba36d08d437237098cd9ac198657e1c9980d243b0e82b1ee65fd850'))

def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, label):
    require(type(value) is int or isinstance(value, np.integer), f'Integer {label} required')
    require(not isinstance(value, (bool, np.bool_)), f'Boolean {label} forbidden')
    return int(value)


def array(value, shape, label, *, positive=False, index=False):
    value=np.asarray(value)
    require(value.shape==shape and value.dtype.kind in 'iuf' and np.isfinite(value).all(), f'Invalid {label}')
    if index:
        require(np.equal(value,np.floor(value)).all(), f'Integer {label} required')
    if positive:
        require((value>0).all(), f'Positive {label} required')
    return value.astype(np.int64 if index else np.float64,copy=True)


def plain(value):
    if isinstance(value,Mapping):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    require(isinstance(value,(str,int,float,bool,type(None))), 'JSON-compatible provenance required')
    return value


class RetainedTraceCollector:
    """One ordered polling-Gibbs event stream; copied immutable final export.

    No-poll direct-prior records lack conditional missing-race locations/scales
    and are deliberately unsupported; archived fundamentals are separate.
    """
    def __init__(self, *, chains, draws, race_ids, observed_race_indices, missing_race_indices,
                 posterior_source_chain, posterior_source_draw, source_identity,
                 source_pins, batch_length=100, weight_sum_tolerance=1e-10):
        self.chains=integer(chains,'chains');self.draws=integer(draws,'draws')
        require(self.chains>=2 and self.draws>=1,'At least two chains and positive retained draws required')
        self.race_ids=tuple(race_ids);self.n=len(self.race_ids)
        require(self.n>0 and len(set(self.race_ids))==self.n and all(isinstance(s,str) and s for s in self.race_ids),'Unique named races required')
        self.observed=array(observed_race_indices,(len(observed_race_indices),),'observed axes',index=True)
        self.missing=array(missing_race_indices,(len(missing_race_indices),),'missing axes',index=True)
        self.m=len(self.observed);self.u=len(self.missing)
        require(self.m>0,'Polling trace requires at least one observed race; direct prior unsupported')
        require(np.array_equal(np.sort(np.r_[self.observed,self.missing]),np.arange(self.n)),'Observed/missing axes must partition races')
        require(np.array_equal(self.observed,np.sort(self.observed)) and np.array_equal(self.missing,np.sort(self.missing)),'Native sorted observed/missing axes required')
        self.k=len(posterior_source_chain)
        require(self.k>0,'Nonempty bank map required')
        self.source_chain=array(posterior_source_chain,(self.k,),'source chain',index=True)
        self.source_draw=array(posterior_source_draw,(self.k,),'source draw',index=True)
        require((self.source_chain>=0).all() and (self.source_draw>=0).all(),'Nonnegative source coordinates required')
        require(len(set(zip(self.source_chain,self.source_draw)))==self.k,'Unique source chain/draw pairs required')
        require(isinstance(source_identity,str) and bool(source_identity),'Source identity required')
        require(isinstance(source_pins,Mapping) and bool(source_pins),'Nonempty source pins required')
        require(all(isinstance(k,str) and isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for k,v in source_pins.items()),'SHA256 source pins required')
        require(all(source_pins.get(k)==v for k,v in REQUIRED_SOURCE_PINS.items()),'Required selected source/design pins mismatch')
        self.source_identity=source_identity;self.source_pins=dict(source_pins)
        require(np.isfinite(weight_sum_tolerance) and 0<weight_sum_tolerance<=1e-6,'Explicit small positive weight tolerance required')
        self.weight_sum_tolerance=float(weight_sum_tolerance)
        self.accumulator=RetainedRowWeightAccumulator(batch_length=batch_length)
        c,d,m,u=self.chains,self.draws,self.m,self.u
        self.values={
            'conditioning_row_index':np.empty((c,d),dtype=np.int64),
            'conditioning_lambda':np.empty((c,d,m)),
            'gaussian_mean_theta_cycle':np.empty((c,d,m+1)),
            'gaussian_variance_theta_cycle':np.empty((c,d,m+1)),
            'observed_theta':np.empty((c,d,m)),
            'sampled_row_index':np.empty((c,d),dtype=np.int64),
            'lambda_after_gamma':np.empty((c,d,m)),
            'cycle':np.empty((c,d)),
            'missing_location':np.empty((c,d,u)),
            'missing_scale':np.empty((c,d)),
        }
        self.chain=0;self.draw=0;self.phase='gaussian_conditional';self.final_states=[]

    def __call__(self, record):
        require(isinstance(record,Mapping),'Mapping observer record required')
        require(self.chain<self.chains,'Trace already complete; duplicate/extra event')
        require(record.get('phase')==self.phase,'Missing/duplicate/reordered observer phase')
        c=integer(record.get('chain'),'record chain')
        require(c==self.chain,'Reordered forecast chain')
        if self.phase=='chain_final_rng':
            require(integer(record.get('retained_draws'),'retained draws')==self.draws,'Wrong retained count')
            state=record.get('state')
            require(isinstance(state,Mapping) and isinstance(state.get('bit_generator'),str) and isinstance(state.get('state'),Mapping),'Generator state provenance required')
            encoded=json.dumps(plain(state),allow_nan=False)
            self.accumulator(record)
            self.final_states.append(json.loads(encoded));self.chain+=1;self.draw=0;self.phase='gaussian_conditional'
            return
        d=integer(record.get('retained_draw'),'retained draw')
        require(d==self.draw,'Missing/duplicate/reordered retained draw')
        if self.phase=='gaussian_conditional':
            row=integer(record.get('conditioning_row_index'),'conditioning row')
            require(0<=row<self.k,'Conditioning row outside bank')
            lam=array(record.get('conditioning_lambda'),(self.m,),'conditioning precision',positive=True)
            mean=array(record.get('mean_theta_cycle'),(self.m+1,),'Gaussian mean')
            variance=array(record.get('marginal_variances'),(self.m+1,),'Gaussian variance',positive=True)
            if d:
                require(row==self.values['sampled_row_index'][c,d-1],'Pre-row differs from preceding completed row')
                require(np.array_equal(lam,self.values['lambda_after_gamma'][c,d-1]),'Pre-lambda differs from preceding post-gamma lambda')
            self.values['conditioning_row_index'][c,d]=row
            self.values['conditioning_lambda'][c,d]=lam
            self.values['gaussian_mean_theta_cycle'][c,d]=mean
            self.values['gaussian_variance_theta_cycle'][c,d]=variance
            self.phase='row_weights'
        elif self.phase=='row_weights':
            row=integer(record.get('conditioning_row_index'),'weight conditioning row')
            lam=array(record.get('conditioning_lambda'),(self.m,),'weight conditioning precision',positive=True)
            theta=array(record.get('theta'),(self.m,),'observed theta')
            cycle=float(record.get('cycle'))
            weights=array(record.get('normalized_row_weights'),(self.k,),'row weights')
            require(np.isfinite(cycle),'Finite cycle required')
            require((weights>=0).all() and abs(weights.sum()-1)<=self.weight_sum_tolerance,'Normalized nonnegative all-bank weights required')
            require(row==self.values['conditioning_row_index'][c,d] and np.array_equal(lam,self.values['conditioning_lambda'][c,d]),'Row-weight conditioning phase mismatch')
            self.accumulator(record)
            self.values['observed_theta'][c,d]=theta;self.values['cycle'][c,d]=cycle
            self.phase='completed_sweep'
        elif self.phase=='completed_sweep':
            row=integer(record.get('sampled_row_index'),'sampled row')
            require(0<=row<self.k,'Sampled row outside bank')
            lam=array(record.get('lambda_after_gamma'),(self.m,),'post-gamma precision',positive=True)
            location=array(record.get('missing_location'),(self.u,),'missing location')
            scale=float(record.get('missing_scale'));cycle=float(record.get('cycle'))
            observed=array(record.get('observed_race_indices'),(self.m,),'completed observed axes',index=True)
            missing=array(record.get('missing_race_indices'),(self.u,),'completed missing axes',index=True)
            require(np.array_equal(observed,self.observed) and np.array_equal(missing,self.missing),'Completed race axes differ')
            require(np.isfinite(scale) and scale>0 and np.isfinite(cycle),'Positive missing scale / finite cycle required')
            require(cycle==self.values['cycle'][c,d],'Completed cycle phase mismatch')
            self.accumulator(record)
            self.values['sampled_row_index'][c,d]=row;self.values['lambda_after_gamma'][c,d]=lam
            self.values['missing_location'][c,d]=location;self.values['missing_scale'][c,d]=scale
            self.draw+=1
            self.phase='chain_final_rng' if self.draw==self.draws else 'gaussian_conditional'

    def export(self):
        require(self.chain==self.chains,'Incomplete observer trace')
        aggregate=self.accumulator.export();length=self.accumulator.batch_length
        batches=(self.draws+length-1)//length
        sums=np.empty((self.chains,batches,self.k))
        counts=np.empty((self.chains,batches),dtype=np.int64)
        occupancy=np.zeros((self.chains,self.k),dtype=np.int64)
        for c in range(self.chains):
            require(aggregate['counts'][c]==self.draws,'Accumulator retention count mismatch')
            for b in range(batches):
                sums[c,b]=aggregate['batch_weight_sums'][(c,b)]
                counts[c,b]=min(length,self.draws-b*length)
            for row,count in aggregate['occupancy'][c].items():occupancy[c,row]=count
        arrays={**self.values,'observed_race_indices':self.observed,'missing_race_indices':self.missing,
                'posterior_source_chain':self.source_chain,'posterior_source_draw':self.source_draw,
                'bank_batch_weight_sums':sums,'bank_batch_counts':counts,'bank_sampled_occupancy':occupancy}
        metadata=dict(schema='RetainedPollingTraceV1',chains=self.chains,draws=self.draws,race_ids=self.race_ids,
                      source_identity=self.source_identity,source_pins=self.source_pins,batch_length=length,
                      final_rng_states=self.final_states,weight_sum_tolerance=self.weight_sum_tolerance,
                      phase_contract='pre_state_row_lambda_gaussian__post_row_gamma__v1')
        return _observed_copy(dict(arrays=arrays,metadata=metadata))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_trace(exported, destination):
    """Exclusive NPZ+JSON write; all NPZ arrays have non-object numeric dtype."""
    directory=Path(destination)
    arrays=dict(exported['arrays']);metadata=plain(exported['metadata'])
    require(all(np.asarray(v).dtype.kind in 'iuf' and not np.asarray(v).dtype.hasobject for v in arrays.values()),'Numeric non-pickle arrays required')
    validate_trace_arrays(arrays,metadata)
    encoded=json.dumps(metadata,allow_nan=False,sort_keys=True,indent=2)+'\n'
    directory.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(directory/'trace.npz',**arrays)
    (directory/'provenance.json').write_text(encoded)
    receipt=dict(schema='RetainedPollingTraceReceiptV1',files={name:sha(directory/name) for name in ['trace.npz','provenance.json']},
                 arrays={name:dict(shape=list(v.shape),dtype=v.dtype.str) for name,v in arrays.items()})
    (directory/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    return _observed_copy(receipt)


def load_trace(destination, *, expected_source_pins):
    """Verify exclusive receipt and required source pins; never use pickle."""
    directory=Path(destination);receipt=json.loads((directory/'receipt.json').read_text())
    require(receipt.get('schema')=='RetainedPollingTraceReceiptV1','Unknown trace receipt')
    require(set(receipt['files'])=={'trace.npz','provenance.json'},'Exact trace file receipt required')
    for name,digest in receipt['files'].items():require(sha(directory/name)==digest,'Trace byte digest mismatch')
    metadata=json.loads((directory/'provenance.json').read_text())
    require(metadata.get('schema')=='RetainedPollingTraceV1','Unknown trace schema')
    require(metadata.get('source_pins')==dict(expected_source_pins),'Trace source pins mismatch')
    with np.load(directory/'trace.npz',allow_pickle=False) as archive:
        require(set(archive.files)==set(receipt['arrays']),'Trace array set mismatch')
        arrays={name:archive[name].copy() for name in archive.files}
    for name,value in arrays.items():
        require(value.dtype.kind in 'iuf' and not value.dtype.hasobject and np.isfinite(value).all(),'Invalid serialized numeric trace')
        require(list(value.shape)==receipt['arrays'][name]['shape'] and value.dtype.str==receipt['arrays'][name]['dtype'],'Trace shape/dtype receipt mismatch')
    validate_trace_arrays(arrays,metadata)
    return _observed_copy(dict(arrays=arrays,metadata=metadata))


def validate_trace_arrays(arrays,metadata):
    require(metadata.get('schema')=='RetainedPollingTraceV1','Unknown trace schema')
    require(metadata.get('phase_contract')=='pre_state_row_lambda_gaussian__post_row_gamma__v1','Serialized phase contract mismatch')
    pins=metadata.get('source_pins',{})
    require(all(pins.get(k)==v for k,v in REQUIRED_SOURCE_PINS.items()),'Serialized required source/design pins mismatch')
    require(isinstance(metadata.get('source_identity'),str) and bool(metadata['source_identity']),'Serialized source identity required')
    c=integer(metadata.get('chains'),'serialized chains');d=integer(metadata.get('draws'),'serialized draws')
    races=tuple(metadata.get('race_ids',()));n=len(races)
    require(c>=2 and d>0 and n>0 and len(set(races))==n,'Invalid serialized dimensions/races')
    observed=np.asarray(arrays.get('observed_race_indices'));missing=np.asarray(arrays.get('missing_race_indices'))
    require(observed.ndim==missing.ndim==1 and len(observed)>0,'Serialized polling axes required')
    require(np.array_equal(np.sort(np.r_[observed,missing]),np.arange(n)),'Serialized axes do not partition races')
    require(np.array_equal(observed,np.sort(observed)) and np.array_equal(missing,np.sort(missing)),'Serialized native sorted axes required')
    m=len(observed);u=len(missing);k=len(arrays.get('posterior_source_chain',()))
    length=integer(metadata.get('batch_length'),'serialized batch length');require(length>0 and k>0,'Positive serialized batch/bank size required')
    b=(d+length-1)//length
    shapes={'conditioning_row_index':(c,d),'conditioning_lambda':(c,d,m),'gaussian_mean_theta_cycle':(c,d,m+1),
            'gaussian_variance_theta_cycle':(c,d,m+1),'observed_theta':(c,d,m),'sampled_row_index':(c,d),
            'lambda_after_gamma':(c,d,m),'cycle':(c,d),'missing_location':(c,d,u),'missing_scale':(c,d),
            'observed_race_indices':(m,),'missing_race_indices':(u,),'posterior_source_chain':(k,),
            'posterior_source_draw':(k,),'bank_batch_weight_sums':(c,b,k),'bank_batch_counts':(c,b),'bank_sampled_occupancy':(c,k)}
    require(set(arrays)==set(shapes),'Exact serialized trace array schema required')
    index_names={'conditioning_row_index','sampled_row_index','observed_race_indices','missing_race_indices','posterior_source_chain','posterior_source_draw','bank_batch_counts','bank_sampled_occupancy'}
    for name,shape in shapes.items():
        value=np.asarray(arrays[name]);require(value.shape==shape and value.dtype.kind in 'iuf' and np.isfinite(value).all(),'Serialized trace shape/value mismatch: '+name)
        if name in index_names:require(value.dtype.kind in 'iu' and (value>=0).all(),'Serialized integer coordinate required: '+name)
    for name in ['conditioning_lambda','lambda_after_gamma','gaussian_variance_theta_cycle','missing_scale']:require((arrays[name]>0).all(),'Serialized positive state required: '+name)
    for name in ['conditioning_row_index','sampled_row_index']:require((arrays[name]<k).all(),'Serialized bank row outside support')
    require(len(set(zip(arrays['posterior_source_chain'],arrays['posterior_source_draw'])))==k,'Serialized duplicate source identity')
    require(np.array_equal(arrays['conditioning_row_index'][:,1:],arrays['sampled_row_index'][:,:-1]) and np.array_equal(arrays['conditioning_lambda'][:,1:],arrays['lambda_after_gamma'][:,:-1]),'Serialized pre/post phase alignment mismatch')
    expected=np.array([min(length,d-j*length) for j in range(b)])
    require(np.array_equal(arrays['bank_batch_counts'],np.tile(expected,(c,1))),'Serialized bank batch retention mismatch')
    require((arrays['bank_batch_weight_sums']>=0).all() and np.allclose(arrays['bank_batch_weight_sums'].sum(axis=2),arrays['bank_batch_counts'],atol=1e-8,rtol=1e-10),'Serialized bank mass mismatch')
    require(np.array_equal(arrays['bank_sampled_occupancy'].sum(axis=1),np.full(c,d)),'Serialized occupancy count mismatch')
    require(len(metadata.get('final_rng_states',()))==c,'Serialized final RNG state count mismatch')
    json.dumps(plain(metadata),allow_nan=False)


@dataclass(frozen=True)
class DiagnosticConfig:
    """Reviewable proposed gates, not an execution or promotion authorization."""
    design_sha256:str=ROOT_DESIGN_SHA256
    batch_lengths:tuple=(50,100)
    max_rhat:float=1.01
    min_mean_ess:float=400.
    max_mean_share_mcse:float=.001
    max_win_absolute_mcse:float=.0025
    max_interval_cdf_mcse:float=.0025
    max_interval_bracket_width:float=.005
    max_tail_relative_mcse:float=.05
    max_chain_logtail_difference:float=.15
    max_bank_chain_tv:float=.05
    max_source_chain_mass_difference:float=.02
    bank_warning_max_row_mass:float=.02
    bank_warning_min_effective_rows:float=400.
    bank_warning_min_99_mass_rows:int=100
    bank_warning_max_source_chain_mass:float=.80

    def __post_init__(self):
        require(self.design_sha256==ROOT_DESIGN_SHA256,'Selected design SHA mismatch')
        require(type(self.batch_lengths) is tuple and tuple(self.batch_lengths)==(50,100),'This diagnostic version requires batch passes50 and100')
        for key,value in asdict(self).items():
            if key not in ('batch_lengths','design_sha256'):require(np.isfinite(value) and value>0,f'Positive finite diagnostic threshold {key} required')
        require(self.max_rhat>=1,'Rhat threshold must be at least1')


def _diagnostic_inputs(values,names,structural_constants):
    values=np.asarray(values,float);names=tuple(names);structural=set(structural_constants)
    require(values.ndim==3 and values.shape[2]==len(names) and len(names)>0 and len(set(names))==len(names),'Unique aligned function coordinates required')
    require(np.isfinite(values).all(),'Finite function traces required')
    require(structural.issubset(names),'Unknown declared structural coordinate')
    constants=np.ptp(values.reshape(-1,len(names)),axis=0)==0
    require(all(constants[j] for j,name in enumerate(names) if name in structural),'Structural declaration requires an actually constant coordinate')
    return values,names,structural,constants


def function_diagnostics(values,names,*,structural_constants=(),max_mean_mcse=None,config=DiagnosticConfig()):
    values,names,structural,constants=_diagnostic_inputs(values,names,structural_constants)
    passes=[correlated_mcse(values,batch_length=b) for b in config.batch_lengths]
    mcse=np.max([p['mcse'] for p in passes],axis=0);rhat=np.max([p['rank_rhat'] for p in passes],axis=0)
    variance=np.var(values.reshape(-1,len(names)),axis=0,ddof=1)
    if max_mean_mcse is not None:
        limits=np.broadcast_to(np.asarray(max_mean_mcse,float),(len(names),))
        require(np.isfinite(limits).all() and (limits>0).all(),'Explicit positive coordinate MCSE limits required')
    else:limits=None
    rows=[]
    for j,name in enumerate(names):
        constant=bool(constants[j]);known=name in structural
        unresolved=(constant and not known) or (not constant and mcse[j]<=0)
        ess=None if constant or mcse[j]<=0 else float(variance[j]/mcse[j]**2)
        gates=dict(rhat=None if unresolved or known else bool(rhat[j]<=config.max_rhat),
                   mean_ess=None if constant or ess is None else bool(ess>=config.min_mean_ess))
        if limits is not None:gates['mean_mcse']=None if unresolved else bool(mcse[j]<=limits[j])
        rows.append(dict(coordinate=name,constant=constant,structural_constant=known,
                         status='structural' if known else 'unresolved_constant' if constant else 'unresolved_zero_mcse' if unresolved else 'measured',
                         mean=float(values[:,:,j].mean()),variance=float(variance[j]),
                         mcse=None if unresolved else float(mcse[j]),
                         rank_folded_split_rhat=None if unresolved or not np.isfinite(rhat[j]) else float(rhat[j]),
                         rhat_nonfinite=bool(not np.isfinite(rhat[j])),mean_ess_estimate=ess,
                         gates=gates,failures=[k for k,v in gates.items() if v is False]))
    return dict(method='existing_rank_folded_split_rhat__maximum_batch_chain_mcse_50_100',
                mean_ess_is_not_bulk_or_tail_ess=True,config=asdict(config),coordinates=rows,
                inherited_bank_error='Not included; extra forecast draws cannot increase historical bank ESS')


def log_tail_diagnostics(log_probabilities,names,*,structural_constants=(),config=DiagnosticConfig()):
    logs=np.asarray(log_probabilities,float);names=tuple(names)
    require(logs.ndim==3 and logs.shape[2]==len(names) and len(set(names))==len(names),'Aligned named log-tail traces required')
    require(not np.isnan(logs).any() and not np.isposinf(logs).any() and (logs<=0).all(),'Nonpositive log probabilities required')
    structural=set(structural_constants);require(structural.issubset(names),'Unknown structural tail coordinate')
    maxima=np.max(logs,axis=(0,1));valid=np.isfinite(maxima)
    scaled=np.zeros_like(logs)
    scaled[:,:,valid]=np.exp(logs[:,:,valid]-maxima[valid])
    # All-minus-infinity coordinates stay unresolved; no invented probability floor.
    basic=function_diagnostics(scaled,names,structural_constants=structural_constants,config=config)
    rows=[]
    for j,name in enumerate(names):
        base=basic['coordinates'][j]
        if not valid[j]:
            require(name not in structural,'All-minus-infinity tail cannot be a structural zero-error certificate')
            rows.append(dict(coordinate=name,status='unresolved_all_negative_infinity',log_probability=None,
                             relative_mcse=None,log_absolute_mcse=None,per_chain_log_difference=None,
                             gates=dict(absolute_mcse=None,relative_mcse=None,chain_log_difference=None),failures=[]))
            continue
        centered_logs=logs[:,:,j]-maxima[j]
        centered_logmean=float(logsumexp(centered_logs)-np.log(logs.shape[0]*logs.shape[1]))
        logmean=float(maxima[j]+centered_logmean)
        differences=logsumexp(centered_logs,axis=1)-np.log(logs.shape[1])-centered_logmean
        mean_scaled=float(scaled[:,:,j].mean());mcse=base['mcse']
        relative=None if mcse is None else float(mcse/mean_scaled)
        log_mcse=None if mcse is None or mcse==0 else float(maxima[j]+np.log(mcse))
        unresolved=base['status'].startswith('unresolved')
        absolute_gate=None if unresolved else (True if mcse==0 else bool(log_mcse<=np.log(config.max_win_absolute_mcse)))
        gates=dict(absolute_mcse=absolute_gate,relative_mcse=None if unresolved else bool(relative<=config.max_tail_relative_mcse),
                   chain_log_difference=None if unresolved else bool(np.max(np.abs(differences))<=config.max_chain_logtail_difference))
        rows.append(dict(coordinate=name,status=base['status'],log_probability=logmean,
                         relative_mcse=relative,log_absolute_mcse=log_mcse,
                         absolute_mcse=None if unresolved or (log_mcse is not None and np.exp(log_mcse)==0) else 0. if mcse==0 else float(np.exp(log_mcse)),
                         absolute_mcse_underflows=bool(log_mcse is not None and np.exp(log_mcse)==0),
                         per_chain_log_difference=[float(x) if np.isfinite(x) else None for x in differences],
                         chains_without_tail_support=np.flatnonzero(~np.isfinite(differences)).tolist(),
                         rank_folded_split_rhat=base['rank_folded_split_rhat'],mean_ess_estimate=base['mean_ess_estimate'],
                         tail_function_diagnostic_gates=base['gates'],gates=gates,failures=[k for k,v in gates.items() if v is False]))
    return dict(method='rescaled_conditional_probability_log_tail_mcse',sides_independent_no_subtraction=True,
                no_probability_floor=True,config=asdict(config),coordinates=rows)


def raw_indicator_diagnostics(indicators,names,*,config=DiagnosticConfig()):
    values=np.asarray(indicators)
    require(np.isin(values,[0,1]).all(),'Binary raw indicators required')
    result=function_diagnostics(values.astype(float),names,config=config)
    result['constant_raw_indicator_is_not_zero_error_certificate']=True
    return result


def bank_diagnostics(batch_weight_sums,batch_counts,posterior_source_chain,*,config=DiagnosticConfig()):
    sums=np.asarray(batch_weight_sums,float);counts=np.asarray(batch_counts)
    require(sums.ndim==3 and counts.shape==sums.shape[:2] and sums.shape[0]>=2 and sums.shape[2]>0,'Aligned bank batch sums/counts required')
    require(np.isfinite(sums).all() and (sums>=0).all() and np.isfinite(counts).all() and (counts>0).all() and np.equal(counts,np.floor(counts)).all(),'Finite nonnegative bank sums and positive integer counts required')
    require(np.allclose(sums.sum(axis=2),counts,atol=1e-8,rtol=1e-10),'Bank mass does not match retained counts')
    source=np.asarray(posterior_source_chain)
    require(source.shape==(sums.shape[2],) and np.isfinite(source).all() and np.equal(source,np.floor(source)).all() and (source>=0).all(),'Original source-chain map required')
    totals=counts.sum(axis=1);chain_weights=sums.sum(axis=1)/totals[:,None]
    weights=sums.sum(axis=(0,1))/counts.sum()
    ordered=np.sort(weights)[::-1]
    support=int(np.searchsorted(np.cumsum(ordered),.99,side='left')+1)
    source_ids=np.unique(source);source_mass=np.array([weights[source==s].sum() for s in source_ids])
    forecast_source_mass=np.array([[w[source==s].sum() for s in source_ids] for w in chain_weights])
    tv=.5*np.abs(chain_weights-weights).sum(axis=1)
    source_difference=np.max(np.abs(forecast_source_mass-source_mass),axis=1)
    effective=float(1/np.sum(weights*weights))
    positive=weights>0;entropy=float(-np.sum(weights[positive]*np.log(weights[positive])))
    gates=dict(forecast_chain_tv=bool(np.max(tv)<=config.max_bank_chain_tv),source_chain_mass_difference=bool(np.max(source_difference)<=config.max_source_chain_mass_difference))
    warnings=dict(max_row_mass=bool(ordered[0]>config.bank_warning_max_row_mass),effective_rows=bool(effective<config.bank_warning_min_effective_rows),support_99=bool(support<config.bank_warning_min_99_mass_rows),source_chain_concentration=bool(source_mass.max()>config.bank_warning_max_source_chain_mass))
    return dict(method='retained_conditional_row_weight_averages',row_mass=weights.tolist(),
                max_row_mass=float(ordered[0]),top10_mass=float(ordered[:10].sum()),top100_mass=float(ordered[:100].sum()),
                entropy=entropy,effective_bank_rows=effective,rows_carrying_99_mass=support,
                original_source_chain_ids=source_ids.tolist(),original_source_chain_mass=source_mass.tolist(),
                forecast_chain_source_mass=forecast_source_mass.tolist(),forecast_chain_tv=tv.tolist(),
                source_chain_mass_difference=source_difference.tolist(),retained_counts=totals.tolist(),
                gates=gates,failures=[k for k,v in gates.items() if not v],bank_warning_flags=warnings,
                bank_approximation_sensitive=any(warnings.values()),bank_effective_rows_is_not_historical_or_gibbs_ess=True,
                inherited_bank_error='Separate from new Gibbs integration error; no uniform source-chain mass requirement',config=asdict(config))


def core_state_diagnostics(result,trace,*,structural_constants=(),config=DiagnosticConfig()):
    """Selected F0 core coordinates, preserving forecast chain/draw axes."""
    arrays=trace['arrays'];metadata=trace['metadata'];obs=arrays['observed_race_indices']
    require(tuple(result['race_ids'])==tuple(metadata['race_ids']),'Original/trace race identity mismatch')
    require(np.array_equal(result['posterior_index'],arrays['sampled_row_index']) and np.array_equal(result['shared_cycle'],arrays['cycle']),'Original/trace row-cycle mismatch')
    require(np.array_equal(result['logits'][:,:,obs],arrays['observed_theta']),'Original/trace observed theta mismatch')
    coefficient_names=tuple(result['coefficient_names'])
    require(len(coefficient_names)==6 and result['coefficients'].shape[-1]==6,'Selected six-column F0 coefficient trace required')
    values=np.concatenate([result['logits'][:,:,obs],result['shares'][:,:,obs],result['coefficients'],
        result['tau_cycle'][...,None],result['tau_race'][...,None],result['shared_cycle'][...,None],
        result['shared_cycle_z'][...,None],np.log(arrays['lambda_after_gamma'])],axis=2)
    race_names=[result['race_ids'][j] for j in obs]
    names=([f'logit:{r}' for r in race_names]+[f'share:{r}' for r in race_names]
           +[f'coefficient:{r}' for r in coefficient_names]+['tau_cycle','tau_race','cycle','cycle_z']
           +[f'log_post_gamma_lambda:{r}' for r in race_names])
    return function_diagnostics(values,names,structural_constants=structural_constants,config=config)


def mean_share_diagnostics(conditional_expected_shares,names,*,structural_constants=(),config=DiagnosticConfig()):
    values=np.asarray(conditional_expected_shares,float)
    require(((values>=0)&(values<=1)).all(),'Conditional expected shares must lie in[0,1]')
    return function_diagnostics(values,names,structural_constants=structural_constants,
                                max_mean_mcse=config.max_mean_share_mcse,config=config)


def interval_display_diagnostics(conditional_cdfs,names,share_bracket_widths,*,structural_constants=(),config=DiagnosticConfig()):
    """Display-only supplied CDF/bracket checks, never score qualification."""
    values=np.asarray(conditional_cdfs,float)
    require(((values>=0)&(values<=1)).all(),'Conditional CDFs must lie in[0,1]')
    widths=array(share_bracket_widths,(len(names),),'supplied share bracket widths')
    require((widths>=0).all(),'Nonnegative share brackets required')
    result=function_diagnostics(values,names,structural_constants=structural_constants,
                                max_mean_mcse=config.max_interval_cdf_mcse,config=config)
    result['scope']='interval_display_only_not_score_precision_or_sampling_retry'
    for row,width in zip(result['coordinates'],widths):
        row['share_bracket_width']=float(width)
        row['gates']['share_bracket_width']=bool(width<=config.max_interval_bracket_width)
        if width>config.max_interval_bracket_width:row['failures'].append('share_bracket_width')
    return result
