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

def completed(out,retained_ack,ordinal,phase):
 out=Path(out);need(type(ordinal)is int and ordinal in range(2,8)and phase in ['dynamic-witness','fixed-selected']and out.name==phase,'completed parent/phase required');need(sha(out/'controller-success.json')==retained_ack,'caller-retained completion digest mismatch');receipt=json.loads((out/'controller-success.json').read_text());need(receipt['status']=='successfully_reaped_selected_phase_and_passed_comparison','partial controller receipt');need(receipt['selected_parent_ordinal']==ordinal and receipt['phase']==phase,'completed receipt parent/phase mismatch');need(receipt['contract_sha256']==EXPECTED_CONTRACT,'completed source changed')
 for name,key in [('result.json','result'),('terminal.json','terminal')]:need(pin(out/name)==receipt[key],'completed result/terminal changed')
 result=json.loads((out/'result.json').read_text());need(result['status']=='completed_calls_not_normalized_conditionals','partial result cannot replay');need(result['selected_parent_ordinal']==ordinal and result['phase']==phase and result['contract_sha256']==EXPECTED_CONTRACT,'completed result selection mismatch');terminal=json.loads((out/'terminal.json').read_text());need(terminal['arm']==phase,'terminal phase mismatch');need(terminal['returncode']==0 and terminal['reason']is None,'partial/failed attempt cannot replay as complete')
 for path,expected in receipt['comparison_artifacts'].items():need(pin(out/path)==expected,'completed comparison changed')
 return receipt

def authenticate_predecessors(c):
 for phase,item in c['predecessor_acknowledgments'].items():
  path=Path(item['caller_return']);need(pin(path)==item['caller_return_pin'],'frozen caller acknowledgment changed');outer=json.loads(path.read_text());tool=outer['write_stdin_return'];need(tool['exit_code']==0,'prior controller tool did not complete');ack=json.loads(tool['output']);need(ack['arm']==phase and ack['status']=='completed','prior acknowledgment phase mismatch');root=Path(item['phase_directory']);need(pin(root/'controller-success.json')==ack['controller_success'],'prior retained controller acknowledgment mismatch');r=json.loads((root/'controller-success.json').read_text());need(pin(root/'result.json')==r['result']and pin(root/'terminal.json')==r['terminal'],'prior result/terminal drift');terminal=json.loads((root/'terminal.json').read_text());need(terminal['returncode']==0 and terminal['reason']is None,'prior phase not cleanly reaped')
  for role,expected in r['comparison_artifacts'].items():need(pin(Path(item['comparison_paths'][role]))==expected,'prior comparison artifact drift')
 return True
