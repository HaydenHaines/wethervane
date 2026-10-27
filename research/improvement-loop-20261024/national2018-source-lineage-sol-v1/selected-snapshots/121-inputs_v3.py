"""Exact new reporting target; reuse old validation only as declared structural projection."""
import copy
import numpy as np
from .dependencies import legacy_inputs as old, rank_joint
from .emission import validate_channel

SCHEMA='senate_national_mixed_report_problem_v1'
MIXED_POLL_TARGET='source_selected_mixed_named_pair_share_v1'
COMMUNITY_IDS=old.COMMUNITY_IDS
Problem=old.Problem
Parent=old.Parent
ESTIMAND='intended_first_preference_conditional_proxy'
POLL_TARGET='intended_first_preference_reported_named_pair_share'

def structural_projection(value):
    result=copy.deepcopy(value)
    result.pop('ranked_measurement')
    result['schema']=old.SCHEMA
    for block in result['polls']:
        if block['mean_target'] in (POLL_TARGET,MIXED_POLL_TARGET):block['mean_target']='final_expected_named_pair_share'
        block.pop('row_mean_targets',None)
    return old.validate_problem(result)

def auxiliary_rule_projection(value):
    """No likelihood built from this projection; freezes legacy rule/style validation."""
    core=structural_projection(value).to_dict();core.pop('report_channels');core['schema']='senate_count_joint_problem_v1'
    ranked={r['contest_id'] for r in value['ranked_measurement']['ranked']}
    core['surveys']=[]  # Full emission validation already ran; legacy rule-only projection.
    # Whole blocks are omitted only from the rule-validator projection; actual
    # measurement consumes original blocks, covariance and tokens unchanged.
    core['polls']=[]  # Actual core retains every complete block and its covariance.
    from src.senate_count_stack_v1.inputs import validate_problem as validate_core
    pp=validate_core(core)
    ext=dict(schema='RankedRunoffJointExtensionV1',core_problem_identity=pp.identity,cutoff_utc=value['cutoff_utc'],sources=value['ranked_measurement']['sources'],ranked=value['ranked_measurement']['ranked'],runoffs=[])
    return pp,rank_joint.validate_extension(core,ext)

def validate_problem(value):
    if type(value) is Problem:value=value.to_dict()
    if type(value) is not dict or value.get('schema')!=SCHEMA:raise ValueError('Explicit intended-report Problem version required')
    if 'ranked_measurement' not in value:raise ValueError('Ranked report configuration required')
    config=value['ranked_measurement']
    old._keys(config,'schema sources ranked survey_estimands poll_channels')
    if config['schema']!='RankedIntendedReportMeasurementV1' or not config['ranked']:raise ValueError('Finite ranked scope required')
    inner=structural_projection(value)
    ranked={r['contest_id']:r for r in config['ranked']}
    if len(ranked)!=len(config['ranked']):raise ValueError('Duplicate ranked contest')
    # Raw CVR/count capture is deliberately outside this successor boundary.
    if any(r['observations'] for r in ranked.values()):raise ValueError('Raw ranked count observations need separately integrated ownership')
    if any(o['contest_id'] in ranked for o in value['official_counts']):raise ValueError('Auxiliary singleton counts cannot be ranked outcomes')
    surveys={s['id']:s for s in value['surveys'] if s['contest_id'] in ranked}
    if set(config['survey_estimands'])!=set(surveys) or any(v!=ESTIMAND for v in config['survey_estimands'].values()):raise ValueError('Every ranked survey needs explicit intended-first-preference proxy; literal/final reports unsupported')
    channels=list(value['report_channels'].values())
    covered=set()
    for block in value['polls']:
        is_rank=[r['contest_id'] in ranked for r in block['rows']]
        if block['mean_target']==MIXED_POLL_TARGET:
            targets=block.get('row_mean_targets')
            expected={r['observation_id']:POLL_TARGET if r['contest_id'] in ranked else 'final_expected_named_pair_share' for r in block['rows']}
            if type(targets) is not dict or targets!=expected:raise ValueError('Every mixed-block row requires its exact supported source target')
        elif 'row_mean_targets' in block:raise ValueError('Row target map requires mixed-block schema')
        if any(is_rank):
            if block['mean_target']!=MIXED_POLL_TARGET and (not all(is_rank) or block['mean_target']!=POLL_TARGET):raise ValueError('Whole covariance block requires explicit per-row mixed target mapping')
            for row in block['rows']:
                if row['contest_id'] not in ranked:continue
                rid=row['observation_id'];covered.add(rid)
                if rid not in config['poll_channels']:raise ValueError('Source-selected poll report operator missing')
                pc=config['poll_channels'][rid]
                old._keys(pc,'estimand channel selection_weights source_choice_shift first_label second_label denominator')
                if pc['estimand']!=ESTIMAND or pc['denominator']!='reported_named_pair_only':raise ValueError('Explicit intended report pair denominator required')
                ch=validate_channel(pc['channel']);channels.append(ch.config)
                names=tuple(ranked[row['contest_id']]['style']['candidate_ids'])
                if ch.candidates!=names or ch.config['source_id']!=block['source_id']:raise ValueError('Poll source/person axes differ')
                for field,cand in [('first_label','first_candidate_id'),('second_label','second_candidate_id')]:
                    label=pc[field]
                    if label not in ch.labels or [k for k,v in zip(ch.candidates,ch.config['candidate_default_labels']) if v==label]!=[row[cand]]:raise ValueError('Poll pair labels must map exactly to prespecified actual people')
                weights=np.asarray(pc['selection_weights'],float)
                if weights.shape!=(24,12) or not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('Positive fixed source-selection operator on full profiles required')
                old._number(pc['source_choice_shift'])
        elif block['mean_target'] not in ('final_expected_named_pair_share',MIXED_POLL_TARGET):raise ValueError('Unsupported nonranked poll target')
    if set(config['poll_channels'])!=covered:raise ValueError('Unused/unsupported poll report channel')
    errors={}
    sharing={}
    for channel in channels:
        c=validate_channel(channel).config;src=c['source_id'];e=c['epsilon']
        if src in errors and errors[src]!=e:raise ValueError('One error configuration per source across surveys/polls')
        errors[src]=e
        if e['kind']=='beta':
            key=e['sharing_key']
            if key in sharing and sharing[key]!=src:raise ValueError('Different sources cannot alias one error prior')
            sharing[key]=src
    auxiliary_rule_projection(value)
    return old._encode(value,Problem)

def validate_parent(problem,value):
    problem=validate_problem(problem)
    if type(value) is Parent:value=value.to_dict()
    if value['problem_identity']!=problem.identity:raise ValueError('Parent must bind new complete reporting target')
    inner=structural_projection(problem.to_dict());projection=copy.deepcopy(value);projection['problem_identity']=inner.identity
    old.validate_parent(inner,projection)
    return old._encode(value,Parent)
