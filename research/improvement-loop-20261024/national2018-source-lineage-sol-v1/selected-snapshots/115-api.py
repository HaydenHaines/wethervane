"""Checked common core builder + unchanged finite ranked scientific contribution."""
import sys
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from . import graph
from .components import standard_registry, baseline_composition
from .inputs_v3 import validate_problem, validate_parent, auxiliary_rule_projection
from .dependencies import package, PINS

PINS['mixed-count-graph-astra-v1']='0470c26de06142e9267138b8d99547fd3b2ae7c4966062fee58ce34a0669dc09'

def source_inventory():
    here=Path(__file__).resolve().parent;root=here.parents[2]
    paths=[here/n for n in ('package.py','api.py','graph.py','components.py','references.py','emission.py','intended.py','inputs_v3.py','dependencies.py','assemble.py','reader.py')]
    for folder in ('survey-answer-emission-astra-v1','ranked-runoff-graph-astra-v1','mixed-count-graph-astra-v1'):
        paths += list((here.parent/folder).glob('*.py'))
    paths += [root/'src/senate_count_stack_v1'/n for n in ('inputs.py','measurement.py','demographics.py')]
    paths += [here.parent/'cycle-separation-astra-v1/cycle_coordinates.py',here.parent/'candidate-role-hierarchy-astra-v1/roles.py']
    return {str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}

def build_core(problem,parent):
    return graph.compile_model(standard_registry(),baseline_composition(),problem,parent,roles=None)

def compose_ranked_rules(model,problem,parent):
    """No baseline builder called; attach count law to SAME checked utility ports.

    The legacy rule validator sees an explicitly identified auxiliary schema
    projection only. It never constructs the inference core or its likelihood.
    All full-source survey/poll factors remain owned by the new measurement role.
    """
    problem=validate_problem(problem);parent=validate_parent(problem,parent)
    contract=dict(model.joint_contract)
    if contract['problem_identity']!=problem.identity or contract['parent_identity']!=parent.identity or contract['version']!=graph.VERSION:
        raise ValueError('Rank contribution requires the exact new full reporting core/parent')
    if model.composition_record['compiler_and_baseline_dependencies']!=source_inventory():
        raise ValueError('Core source inventory differs')
    core,extension=auxiliary_rule_projection(problem.to_dict())
    protected=parent.to_dict();protected['problem_identity']=core.identity
    mixed=package('mixed-count-graph-astra-v1')
    contribution=sys.modules[mixed.__name__+'.rank_contribution'].contribute(model,core.to_dict(),protected,extension)
    contract.update(scientific_law_version='ranked_intended_report_complete_joint_v1',
        scientific_target_id=graph.digest(dict(core_target=contract['scientific_target_id'],rules=extension,dependencies=source_inventory())),
        required_free_rvs=tuple(v.name for v in model.free_RVs),
        required_physical_diagnostics=tuple(dict.fromkeys(tuple(contract['required_physical_diagnostics'])+tuple(contribution['physical']))),
        ranked_rule_validation_projection=core.identity,
        ranked_rule_projection_used_for_likelihood=False,
        auxiliary_singleton_ranked_outputs_not_tabulation=True,
        prediction_adapter='requires_ranked_report_target_aware_successor',
        runtime_qualified=False,baseline_equivalence_certified=False)
    model.joint_contract=MappingProxyType(contract)
    model.ranked_contribution_inventory=graph.freeze(contribution)
    return model

def build_model(problem,parent,*,core_builder=build_core):
    problem=validate_problem(problem);parent=validate_parent(problem,parent)
    return compose_ranked_rules(core_builder(problem,parent),problem,parent)


def validate_mixed_extension(problem, extension):
    """Validate rule-only projection, then restore full observation-scope checks."""
    problem=validate_problem(problem)
    core,rank=auxiliary_rule_projection(problem.to_dict())
    mixed=package('mixed-count-graph-astra-v1')
    result=mixed.validate_extension(core.to_dict(),extension)
    rr=result['ranked_runoff']
    if rr is None or rr['ranked']!=rank['ranked']:
        raise ValueError('Exactly the report-bound finite ranked styles required')
    actual_sources={r['id']:r for r in rr['sources']}
    if any(actual_sources.get(r['id'])!=r for r in rank['sources']):raise ValueError('Changed reporting rule source identity')
    if any(x['observations'] for x in rr['ranked']+rr['runoffs']):
        raise ValueError('Raw extended observations require additional ownership integration')
    openpart=result['open_person']
    if openpart and any(x['observations'] for x in openpart['contests']):
        raise ValueError('Open count observations not integrated in this successor')
    # Omitting data from the legacy projection must NOT permit mismatched actual
    # observation laws. Only finite-ranked reporting has a selected bridge.
    unsupported={r['runoff_contest_id'] for r in rr['runoffs']}
    if openpart:unsupported|={x['contest_id'] for x in openpart['contests']}
    value=problem.to_dict()
    if any(x['contest_id'] in unsupported for x in value['surveys']+value['official_counts']) or any(r['contest_id'] in unsupported for b in value['polls'] for r in b['rows']):
        raise ValueError('Open-person or conditional-runoff report bridge unselected')
    return core,result


def build_mixed_model(problem,parent,extension,*,core_builder=build_core):
    problem=validate_problem(problem);parent=validate_parent(problem,parent)
    core,extension=validate_mixed_extension(problem,extension)
    model=core_builder(problem,parent);contract=dict(model.joint_contract)
    if contract['problem_identity']!=problem.identity or contract['parent_identity']!=parent.identity or contract['version']!=graph.VERSION:
        raise ValueError('Exact complete reporting core/parent required')
    if model.composition_record['compiler_and_baseline_dependencies']!=source_inventory():raise ValueError('Changed core source inventory')
    mixed=package('mixed-count-graph-astra-v1');protected=parent.to_dict();protected['problem_identity']=core.identity
    stages={'ranked_runoff':sys.modules[mixed.__name__+'.rank_contribution'].contribute(model,core.to_dict(),protected,extension['ranked_runoff'])}
    if extension['open_person'] is not None:stages['open_person']=sys.modules[mixed.__name__+'.open_contribution'].contribute(model,core.to_dict(),protected,extension['open_person'])
    for key in ['parameter_owners','factor_owners']:
        names=[name for stage in stages.values() for name in stage[key]]
        if len(names)!=len(set(names)):raise ValueError('Competing extension ownership')
    contract.update(scientific_law_version='national_report_mixed_joint_assembly_v1',scientific_target_id=graph.digest(dict(core_target=contract['scientific_target_id'],extension=extension,dependencies=source_inventory())),required_free_rvs=tuple(v.name for v in model.free_RVs),required_physical_diagnostics=tuple(dict.fromkeys(tuple(contract['required_physical_diagnostics'])+tuple(n for s in stages.values() for n in s['physical']))),rule_validation_projection=core.identity,rule_projection_used_for_likelihood=False,auxiliary_singleton_extended_outputs_not_tabulation=True,prediction_adapter='requires_national_report_mixed_target_aware_successor',runtime_qualified=False,baseline_equivalence_certified=False)
    model.joint_contract=MappingProxyType(contract);model.national_mixed_contribution_inventory=graph.freeze(stages)
    return model
