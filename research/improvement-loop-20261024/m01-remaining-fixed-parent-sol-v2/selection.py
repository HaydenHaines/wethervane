"""One parent per phase; immutable reservation; completed replay is read-only."""
import json
from pathlib import Path
from common import *
def selected(c,ordinal):
 need(type(ordinal)is int and ordinal in range(2,8),'only one remaining ordinal2..7')
 row=c['remaining_parents'][str(ordinal)];need(row['ordinal']==ordinal,'selection contract mismatch');return row

def reserve(root,c,ordinal,phase):
 selected(c,ordinal);need(phase in ['dynamic-witness','fixed-selected'],'unsupported phase');root=Path(root);root.mkdir(exist_ok=True)
 parent=root/('parent-'+str(ordinal))
 if phase=='dynamic-witness':parent.mkdir(exist_ok=False)
 else:need(parent.is_dir(),'witness phase required before fixed parent')
 out=parent/phase;out.mkdir(exist_ok=False);atomic_json(out/'selected-parent.json',dict(schema='ExclusiveRemainingParentSelectionV1',ordinal=ordinal,parent=c['remaining_parents'][str(ordinal)],phase=phase,contract_sha256=EXPECTED_CONTRACT));return out

def comparison_roster(phase):
 need(phase in ['dynamic-witness','fixed-selected'],'unsupported completion phase')
 return {'comparison-execution.json','../'+('dynamic-witness-comparison.json'if phase=='dynamic-witness'else'fixed-selected-comparison.json')}

def validate_comparisons(out,receipt,ordinal,phase):
 out=Path(out);need(type(receipt['comparison_artifacts'])is dict and set(receipt['comparison_artifacts'])==comparison_roster(phase),'exact per-phase comparison artifact roster required')
 for name,expected in receipt['comparison_artifacts'].items():need(pin(out/name)==expected,'completed comparison changed')
 execution=json.loads((out/'comparison-execution.json').read_text());need(set(execution)=={'schema','selected_parent_ordinal','phase','source_sha256','review_digest','contract_sha256','scope'},'exact comparison execution schema required');need(execution['schema']=='RemainingVerifiedComparisonExecutionV1'and execution['selected_parent_ordinal']==ordinal and execution['phase']==phase and execution['contract_sha256']==EXPECTED_CONTRACT,'comparison execution identity mismatch')
 auth=json.loads((out/'authorization.json').read_text());need(auth['selected_parent_ordinal']==ordinal and auth['contract_sha256']==EXPECTED_CONTRACT,'authorization parent/source mismatch');snapshot=reviewed_sources(auth['review'],auth['review_digest'],EXPECTED_CONTRACT);need(phase in auth['review']['allowed_phases']and ordinal in auth['review']['allowed_parent_ordinals'],'stage not reviewed');need(digest(auth['resource_receipt'])==auth['resources_digest']and auth['resource_receipt']['phase']==phase and auth['resource_receipt']['selected_parent_ordinal']==ordinal and auth['resource_receipt']['contract_sha256']==EXPECTED_CONTRACT,'retained resource identity mismatch')
 need(execution['review_digest']==auth['review_digest']and execution['source_sha256']=={n:__import__('hashlib').sha256(raw).hexdigest()for n,raw in snapshot.items()},'executed comparison source differs from reviewed stage')
 name='../'+('dynamic-witness-comparison.json'if phase=='dynamic-witness'else'fixed-selected-comparison.json');comparison=json.loads((out/name).read_text());expected_schema='RemainingParentDynamicFDWitnessV1'if phase=='dynamic-witness'else'RemainingIndependentFixedParentV1';expected_status='three_saved_dynamic_points_match_new_owned_FD_witness'if phase=='dynamic-witness'else'three_point_factor_rate_physical_and_exact_FD_comparison_passed';need(comparison['schema']==expected_schema and comparison['status']==expected_status and comparison['ordinal']==ordinal and comparison['conditional_normalization']=='unverified','completed numerical comparison schema/status/parent mismatch')
 expected_fields={'schema','ordinal','status','errors','FD_variables','why_new_witness','conditional_normalization'}if phase=='dynamic-witness'else{'schema','ordinal','parent_identity','status','errors','FD_errors','outer_masses','other_remaining_parents_not_inferred','conditional_normalization','scope'}
 need(set(comparison)==expected_fields,'exact phase-specific comparison schema required')
 need(type(comparison['errors'])is list and len(comparison['errors'])==3,'three numerical comparison points required')
 for errors in comparison['errors']:
  need(type(errors)is dict and set(errors)=={'logp','poll_logp','factors','rates','physical'}and all(type(x)in [int,float]and __import__('math').isfinite(x)and x>=0 for x in errors.values()),'exact finite comparison error fields required')
 if phase=='dynamic-witness':need(comparison['FD_variables']==read_contract()['finite_difference_variables']and comparison['why_new_witness']=='original dynamic FD calls existed only for parent1; no derivative fabricated for2..7','exact witness FD variable/provenance required')
 else:need([r['variable']for r in comparison['FD_errors']]==read_contract()['finite_difference_variables']and comparison['outer_masses']==[.125]*8 and comparison['parent_identity']==selected(read_contract(),ordinal)['parent_identity'],'fixed FD/parent/mass mismatch');need(comparison['other_remaining_parents_not_inferred']==[i for i in range(2,8)if i!=ordinal]and comparison['scope']=='one independently selected fixed parent, pointwise only; no sampler/posterior or outer evidence reweight','fixed comparison scope mismatch');need(all(set(row)=={'variable','error'}and type(row['error'])in [int,float]and __import__('math').isfinite(row['error'])and row['error']>=0 for row in comparison['FD_errors']),'exact fixed FD error schema required')
 return receipt['comparison_artifacts']

def controller_response(out,ordinal,phase):
 out=Path(out);receipt=json.loads((out/'controller-success.json').read_text());return dict(schema='RemainingStageControllerReturnV2',phase=phase,selected_parent_ordinal=ordinal,status='completed',contract_sha256=EXPECTED_CONTRACT,controller_success=pin(out/'controller-success.json'),comparisons=receipt['comparison_artifacts'])

def publish_success(out,ordinal,phase,comparisons):
 """Controller-only publication after clean reaping and actual verified comparison."""
 out=Path(out);result=json.loads((out/'result.json').read_text());terminal=json.loads((out/'terminal.json').read_text());need(result['status']=='completed_calls_not_normalized_conditionals'and result['selected_parent_ordinal']==ordinal and result['phase']==phase and result['contract_sha256']==EXPECTED_CONTRACT,'cannot publish partial result');need(terminal['arm']==phase and terminal['returncode']==0 and terminal['reason']is None,'cannot publish failed/unreaped phase');receipt=dict(schema='RemainingStageControllerSuccessV2',result=pin(out/'result.json'),terminal=pin(out/'terminal.json'),authorization=pin(out/'authorization.json'),comparison_artifacts=comparisons,contract_sha256=EXPECTED_CONTRACT,selected_parent_ordinal=ordinal,phase=phase,status='successfully_reaped_selected_phase_and_passed_comparison');validate_comparisons(out,receipt,ordinal,phase);atomic_json(out/'controller-success.json',receipt);return controller_response(out,ordinal,phase)

def completed(out,retained_return_path,retained_return_sha,ordinal,phase):
 out=Path(out);need(type(ordinal)is int and ordinal in range(2,8)and phase in ['dynamic-witness','fixed-selected']and out.name==phase,'completed parent/phase required');path=Path(retained_return_path);need(sha(path)==retained_return_sha,'caller-retained tool-return file changed');retained=json.loads(path.read_text());need(retained['schema']=='RetainedSelectedStageToolReturnV1','retained tool-return schema required');tool=retained['write_stdin_return'];need(type(tool['exit_code'])is int and tool['exit_code']==0 and type(tool['output'])is str,'retained controller tool did not complete successfully');ack=json.loads(tool['output']);need(ack==controller_response(out,ordinal,phase),'retained complete controller response mismatch')
 receipt=json.loads((out/'controller-success.json').read_text());need(set(receipt)=={'schema','status','phase','selected_parent_ordinal','contract_sha256','result','terminal','authorization','comparison_artifacts'},'exact controller success schema required');need(receipt['schema']=='RemainingStageControllerSuccessV2'and receipt['status']=='successfully_reaped_selected_phase_and_passed_comparison','partial controller receipt');need(receipt['selected_parent_ordinal']==ordinal and receipt['phase']==phase and receipt['contract_sha256']==EXPECTED_CONTRACT,'completed receipt selection/source mismatch')
 for name,key in [('result.json','result'),('terminal.json','terminal'),('authorization.json','authorization')]:need(pin(out/name)==receipt[key],'completed result/terminal/authorization changed')
 result=json.loads((out/'result.json').read_text());need(result['status']=='completed_calls_not_normalized_conditionals'and result['selected_parent_ordinal']==ordinal and result['phase']==phase and result['contract_sha256']==EXPECTED_CONTRACT,'completed result mismatch');terminal=json.loads((out/'terminal.json').read_text());need(terminal['arm']==phase and terminal['returncode']==0 and terminal['reason']is None,'partial/failed attempt cannot replay');validate_comparisons(out,receipt,ordinal,phase);return receipt

def authenticate_predecessors(c):
 for phase,item in c['predecessor_acknowledgments'].items():
  path=Path(item['caller_return']);need(pin(path)==item['caller_return_pin'],'frozen caller acknowledgment changed');outer=json.loads(path.read_text());tool=outer['write_stdin_return'];need(tool['exit_code']==0,'prior controller tool did not complete');ack=json.loads(tool['output']);need(ack['arm']==phase and ack['status']=='completed','prior acknowledgment phase mismatch');root=Path(item['phase_directory']);need(pin(root/'controller-success.json')==ack['controller_success'],'prior retained controller acknowledgment mismatch');r=json.loads((root/'controller-success.json').read_text());need(pin(root/'result.json')==r['result']and pin(root/'terminal.json')==r['terminal'],'prior result/terminal drift');terminal=json.loads((root/'terminal.json').read_text());need(terminal['returncode']==0 and terminal['reason']is None,'prior phase not cleanly reaped')
  for role,expected in r['comparison_artifacts'].items():need(pin(Path(item['comparison_paths'][role]))==expected,'prior comparison artifact drift')
 return True
