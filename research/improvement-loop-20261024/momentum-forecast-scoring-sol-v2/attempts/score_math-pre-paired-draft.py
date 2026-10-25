"""Supplied forecast/label score arithmetic, never source loading or sampling."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from marginal import GaussianLogitMixture,weighted_interval_score
from trace_diagnostics import function_diagnostics

LEVELS=(.025,.1,.25,.5,.75,.9,.975)
METRICS=('crps','brier','negative_log','wis')
MAX_CYCLE_MCSE=dict(crps=.0005,brier=.0005,negative_log=.005,wis=.0005)
DESIGN_SHA256='45707d4a933ded2f7d4bffcbaab0fb6fb9a4ab7b1d62ad4494a27422b7d75678'


def require(ok,message):
    if not ok:raise ValueError(message)


def shares(values,label):
    values=np.asarray(values,float)
    require(np.isfinite(values).all() and ((values>=0)&(values<=1)).all(),f'Finite bounded {label} required')
    return values


def centered(values):return values-values.mean(axis=(0,1),keepdims=True)


def finite_score(value,status='computed',reason=None):
    value=float(value)
    return dict(value=value if np.isfinite(value) else None,positive_infinity=bool(np.isposinf(value)),
                status=status,reason=reason)


def empirical_quantiles(values):
    ordered=np.sort(np.asarray(values,float).reshape(-1))
    require(len(ordered)>0,'Nonempty empirical distribution required')
    return {q:float(ordered[int(np.ceil(len(ordered)*q))-1]) for q in LEVELS}


def empirical_crps_and_influence(values,truth):
    """Exact empirical CRPS and centered influence in original supplied order."""
    values=shares(values,'empirical predictions');truth=float(shares(truth,'truth'))
    flat=values.reshape(-1);require(flat.size>0,'Nonempty empirical predictions required')
    order=np.argsort(flat,kind='stable');ordered=flat[order];n=len(flat)
    prefix=np.cumsum(ordered)-ordered
    pair_sorted=((2*np.arange(n)-n)*ordered+ordered.sum()-2*prefix)/n
    pair=np.empty_like(flat);pair[order]=pair_sorted
    score=float(np.mean(abs(flat-truth))-.5*np.mean(pair))
    influence=(abs(flat-truth)-pair).reshape(values.shape)
    influence-=influence.mean()
    return score,influence


def gaussian_race(means,sds,truth,*,win_tail_qualified,lose_tail_qualified):
    """Continuous equal-weight marginal, with original chain/draw components."""
    means=np.asarray(means,float);sds=np.asarray(sds,float)
    require(means.shape==sds.shape and means.ndim==2 and np.isfinite(means).all() and np.isfinite(sds).all() and (sds>0).all(),'Aligned finite chain/draw Gaussian parameters required')
    truth=float(shares(truth,'truth'));model=GaussianLogitMixture(means.ravel(),sds.ravel())
    n=means.size;quantiles={q:model.quantile(q)[0] for q in LEVELS}
    logwin,loglose=model.log_win_lose();component_lose,component_win=model.component_logtails(.5)
    tie=truth==.5;winner=truth>.5
    result=dict(quantiles=quantiles,win_probability=float(np.exp(logwin)),log_win=logwin,log_lose=loglose,
                tie=tie,metrics={},influences={},wis_endpoint_or_density_unresolved=False)
    crps,crps_receipt=model.crps(truth);crps_i,crps_i_receipt=model.crps_influence(truth)
    result['metrics']['crps']=finite_score(crps);result['influences']['crps']=centered(crps_i.reshape(means.shape))
    result['metrics']['wis']=finite_score(weighted_interval_score(truth,quantiles))
    density=np.array([model.density(quantiles[q]) for q in LEVELS])
    stable=bool(np.isfinite(density).all() and (density>0).all())
    wis_i=np.zeros(n)
    one_sided=[]
    if stable:
        for j,q in enumerate(LEVELS):
            logcdf,_=model.component_logtails(quantiles[q])
            q_i=-(np.exp(logcdf)-q)/density[j]
            # Seven pinball losses /3.5 are exactly standard50/80/95 WIS.
            # At a label/quantile kink no unique derivative exists.
            if truth==quantiles[q]:one_sided.append(q)
            wis_i+=(float(truth<quantiles[q])-q)*q_i/3.5
        result['influences']['wis']=centered(wis_i.reshape(means.shape))
    if not stable or one_sided:
        result['wis_endpoint_or_density_unresolved']=True
        result['metrics']['wis']['status']='uncertainty_unresolved'
        result['metrics']['wis']['reason']='one_sided_quantile_kink' if one_sided else 'unstable_quantile_density'
        result['wis_one_sided_quantile_levels']=one_sided
    if tie:
        for metric in ['brier','negative_log']:
            result['metrics'][metric]=dict(value=None,positive_infinity=False,status='not_applicable_tie',reason='exact_binary_tie')
    else:
        component_log_true=component_win if winner else component_lose
        component_log_wrong=component_lose if winner else component_win
        log_true=logwin if winner else loglose;log_wrong=loglose if winner else logwin
        wrong=np.exp(log_wrong)
        # Wrong-side expansion avoids cancellation in1-p for a nearly certain winner.
        wrong_components=np.exp(component_log_wrong).reshape(means.shape)
        brier=float(wrong*wrong)
        result['metrics']['brier']=finite_score(brier)
        result['influences']['brier']=centered(2*wrong*(wrong_components-wrong))
        qualified=bool(win_tail_qualified if winner else lose_tail_qualified)
        result['metrics']['negative_log']=finite_score(-log_true,'computed' if qualified else 'prelabel_true_tail_unqualified',None if qualified else 'true_outcome_side_failed_prelabel_gate')
        # Ratio bounded by total component count; neither exp(log_true) nor1-p required.
        ratio=np.exp(component_log_true-log_true).reshape(means.shape)
        result['influences']['negative_log']=centered(1-ratio)
    result['quadrature']=dict(crps=crps_receipt,crps_influence=crps_i_receipt)
    result['model']=model
    return result


def gaussian_chain_scores(means,sds,truth):
    """Per-chain nonlinear scores, solely to audit the influence approximation."""
    rows={metric:[] for metric in METRICS}
    for c in range(len(means)):
        model=GaussianLogitMixture(means[c],sds[c]);quantiles={q:model.quantile(q)[0] for q in LEVELS}
        rows['crps'].append(model.crps(truth)[0]);rows['wis'].append(weighted_interval_score(truth,quantiles))
        logwin,loglose=model.log_win_lose()
        if truth==.5:
            rows['brier'].append(np.nan);rows['negative_log'].append(np.nan)
        else:
            log_true=logwin if truth>.5 else loglose;log_wrong=loglose if truth>.5 else logwin
            rows['brier'].append(np.exp(2*log_wrong));rows['negative_log'].append(-log_true)
    return rows


def empirical_race(values,truth,*,win_tail_qualified,lose_tail_qualified):
    """Original source-chain,row,5-replicate axis is mandatory for F0 errors."""
    values=shares(values,'archived F0 predictions')
    require(values.ndim==3 and values.shape[2]==5 and values.shape[0]==4,'Original4-source-chain/row/5-replicate F0 axes required')
    truth=float(shares(truth,'truth'));tie=truth==.5
    quantiles=empirical_quantiles(values);p=float((values>.5).mean())
    crps,crps_i=empirical_crps_and_influence(values,truth)
    row_influence=crps_i.mean(axis=2)
    result=dict(quantiles=quantiles,win_probability=p,tie=tie,metrics={},influences={},
                uncertainty_scope='archived original source-row and predictive-replicate sampling; inherited bank dependence included')
    result['metrics']['crps']=finite_score(crps);result['influences']['crps']=centered(row_influence)
    result['metrics']['wis']=finite_score(weighted_interval_score(truth,quantiles),'source_chain_dispersion_only','no_smooth_empirical_density_or_certified_WIS_MCSE')
    chain_wis=[weighted_interval_score(truth,empirical_quantiles(chain)) for chain in values]
    result['wis_source_chain_dispersion']=dict(values=chain_wis,standard_deviation=float(np.std(chain_wis,ddof=1)),certified_mcse=False)
    if tie:
        for metric in ['brier','negative_log']:result['metrics'][metric]=dict(value=None,positive_infinity=False,status='not_applicable_tie',reason='exact_binary_tie')
    else:
        z=float(truth>.5);row_p=(values>.5).mean(axis=2)
        result['metrics']['brier']=finite_score((p-z)**2)
        result['influences']['brier']=centered(2*(p-z)*(row_p-p))
        true_probability=p if truth>.5 else 1-p
        qualified=bool(win_tail_qualified if truth>.5 else lose_tail_qualified)
        if true_probability==0:
            result['metrics']['negative_log']=finite_score(np.inf,'prelabel_true_tail_unqualified','zero_empirical_true_outcome_probability')
        else:
            result['metrics']['negative_log']=finite_score(-np.log(true_probability),'computed' if qualified else 'prelabel_true_tail_unqualified',None if qualified else 'true_outcome_side_failed_prelabel_gate')
            result['influences']['negative_log']=centered(-(row_p-p)/p if truth>.5 else (row_p-p)/(1-p))
    return result


def metric_aggregate(rows,influences,chain_scores,metric,*,deterministic=False,uncertainty_scope='conditional_fixed_bank_Gibbs_integration'):
    """Average races BEFORE MCSE; never drop numerically failed races."""
    selected=[j for j,row in enumerate(rows) if not (row['tie'] and metric in ('brier','negative_log'))]
    if not selected:return dict(value=None,status='empty_binary_or_race_mask',count=0,mcse=None,qualified=False),None
    values=[rows[j]['metrics'][metric]['value'] for j in selected]
    has_infinity=any(rows[j]['metrics'][metric]['positive_infinity'] for j in selected)
    point=None if has_infinity or any(v is None for v in values) else float(np.mean(values))
    unavailable=[j for j in selected if rows[j]['metrics'][metric]['status']!='computed']
    output=dict(value=point,positive_infinity=has_infinity,count=len(selected),mcse=None,
                qualified=False,status='metric_unresolved' if unavailable else 'computed',unqualified_race_indices=unavailable,
                uncertainty_scope=uncertainty_scope,numerical_limit=MAX_CYCLE_MCSE[metric])
    if deterministic and not unavailable and point is not None:
        output.update(qualified=True,status='deterministic_analytic_no_MCMC',mcse=0.)
        return output,None
    if metric=='wis' and uncertainty_scope.startswith('archived'):
        output['status']='source_chain_dispersion_only';return output,None
    if point is None or unavailable or any(influences[j].get(metric) is None for j in selected):return output,None
    average=np.mean(np.stack([influences[j][metric] for j in selected],axis=2),axis=2)
    diagnostic=function_diagnostics(average[...,None],[metric],max_mean_mcse=MAX_CYCLE_MCSE[metric])
    check=diagnostic['coordinates'][0];output['influence_diagnostic']=diagnostic
    output['mcse']=check['mcse']
    if check['mcse'] is None:
        output['status']='uncertainty_unresolved';return output,average
    passed=check['gates']['mean_mcse'] is True
    nonlinear=None
    if chain_scores is not None:
        nonlinear=np.mean(np.stack([chain_scores[j][metric] for j in selected],axis=1),axis=1)
        residual=nonlinear-point-average.mean(axis=1)
        output['nonlinear_chain_check']=dict(scores=nonlinear.tolist(),linearization_residual=residual.tolist(),
            max_residual=float(np.max(abs(residual))),limit=3*check['mcse'],passed=bool(np.max(abs(residual))<=3*check['mcse']))
        passed=passed and output['nonlinear_chain_check']['passed']
    output.update(qualified=bool(passed),status='qualified_conditional_numerical' if passed else 'uncertainty_unresolved')
    return output,average


def score_gaussian_forecast(means,sds,truths,race_ids,win_tail_qualified,lose_tail_qualified,*,deterministic=False):
    means=np.asarray(means,float);sds=np.asarray(sds,float);truths=shares(truths,'truths');race_ids=tuple(race_ids)
    require(means.shape==sds.shape and means.ndim==3 and means.shape[2]==len(race_ids)==len(truths) and len(set(race_ids))==len(race_ids),'Exact aligned Gaussian/race/truth axes required')
    require(len(win_tail_qualified)==len(lose_tail_qualified)==len(race_ids),'Prelabel per-side qualification axes required')
    rows=[];chain=[]
    for j,race in enumerate(race_ids):
        try:
            row=gaussian_race(means[:,:,j],sds[:,:,j],truths[j],win_tail_qualified=win_tail_qualified[j],lose_tail_qualified=lose_tail_qualified[j])
            chain.append(gaussian_chain_scores(means[:,:,j],sds[:,:,j],truths[j]) if not deterministic else None)
        except ArithmeticError as error:
            # Numerical unresolved is preserved; source/axis errors are never caught here.
            row=dict(tie=bool(truths[j]==.5),metrics={metric:dict(value=None,positive_infinity=False,status='deterministic_integration_unresolved',reason=str(error)) for metric in METRICS},influences={})
            chain.append(None)
        row['race_id']=race;row['actual_dem_share']=float(truths[j]);rows.append(row)
    influence=[row['influences'] for row in rows]
    aggregates={};cycle_influence={}
    for metric in METRICS:
        aggregates[metric],cycle_influence[metric]=metric_aggregate(rows,influence,None if deterministic or any(c is None for c in chain) else chain,metric,deterministic=deterministic)
    for row in rows:row.pop('model',None);row.pop('influences',None)
    return dict(rows=rows,aggregates=aggregates,binary_tie_count=int(np.sum(truths==.5)),
                mask_race_ids=list(race_ids),cycle_influences=cycle_influence)


def score_empirical_forecast(values,truths,race_ids,win_tail_qualified,lose_tail_qualified):
    values=shares(values,'archived predictions');truths=shares(truths,'truths');race_ids=tuple(race_ids)
    require(values.ndim==4 and values.shape[2]==5 and values.shape[0]==4 and values.shape[3]==len(race_ids)==len(truths),'Exact4-source-chain/row/5-replicate/race F0 axes required')
    rows=[empirical_race(values[:,:,:,j],truths[j],win_tail_qualified=win_tail_qualified[j],lose_tail_qualified=lose_tail_qualified[j]) for j in range(len(race_ids))]
    influence=[row['influences'] for row in rows];aggregates={};cycle_influence={}
    for metric in METRICS:
        aggregates[metric],cycle_influence[metric]=metric_aggregate(rows,influence,None,metric,uncertainty_scope='archived_original_source_row_and_predictive_simulation')
    for j,row in enumerate(rows):row['race_id']=race_ids[j];row['actual_dem_share']=float(truths[j]);row.pop('influences',None)
    return dict(rows=rows,aggregates=aggregates,binary_tie_count=int(np.sum(truths==.5)),mask_race_ids=list(race_ids),cycle_influences=cycle_influence)


def winner_reference_scores(probability,truths):
    require(np.isfinite(probability) and 0<probability<1,'Proper fixed winner-reference probability required')
    truths=shares(truths,'truths');eligible=truths!=.5
    z=truths[eligible]>.5
    return dict(probability=float(probability),binary_count=int(eligible.sum()),tie_count=int((~eligible).sum()),
                brier=None if not len(z) else float(np.mean((probability-z)**2)),
                negative_log=None if not len(z) else float(np.mean(np.where(z,-np.log(probability),-np.log1p(-probability)))),
                share_distribution=None)


def paired_difference(first,second,metric):
    a,b=first['aggregates'][metric],second['aggregates'][metric]
    require(first['mask_race_ids']==second['mask_race_ids'],'Paired comparisons require exact fixed race masks')
    value=None if a['value'] is None or b['value'] is None else a['value']-b['value']
    independent=all(x['uncertainty_scope']=='conditional_fixed_bank_Gibbs_integration' for x in [a,b])
    mcse=float(np.hypot(a['mcse'],b['mcse'])) if independent and a['qualified'] and b['qualified'] else None
    return dict(value=value,mcse=mcse,three_mcse=None if mcse is None else 3*mcse,
                numerically_resolved=None if mcse is None or value is None else bool(abs(value)>=3*mcse),
                independent_arm_streams=independent,inherited_bank_error_included=False)


def equal_cycle_summary(cycle_reports,metric,*,tier):
    required={'A':(2018,2022),'B':(2020,2024)}[tier]
    if not all(c in cycle_reports for c in required):return dict(status='missing_required_cycles',cycles=list(required),value=None,mcse=None)
    rows=[cycle_reports[c]['aggregates'][metric] for c in required]
    value=None if any(r['value'] is None for r in rows) else float(np.mean([r['value'] for r in rows]))
    qualified=all(r['qualified'] for r in rows)
    mcse=float(np.sqrt(sum(r['mcse']**2 for r in rows))/2) if qualified and all(r['mcse'] is not None for r in rows) else None
    return dict(cycles=list(required),value=value,mcse=mcse,qualified=qualified,
                status='qualified_conditional_numerical' if qualified else 'metric_unresolved',
                election_generalization_confidence_interval=False)
