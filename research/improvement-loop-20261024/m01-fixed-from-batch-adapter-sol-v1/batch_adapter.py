"""Explicit batch-to-one-parent bridge; full batch checks precede selection."""
import hashlib,json,sys,types
from contextlib import contextmanager
from pathlib import Path
from common import *
MODULES=['common','evaluator','selection','baseline_compare','compare','completion','comparison_gate']
@contextmanager
def frozen_batch(c):
    """Absolute-import names are scoped to old contract; restore every alias afterward."""
    source=Path(c['batch_source']);previous={name:sys.modules.get(name)for name in [*MODULES,'compare_parent_binding']};private=[];modules={}
    try:
        for name in MODULES:
            path=source/(name+'.py');raw=path.read_bytes();need(hashlib.sha256(raw).hexdigest()==c['pins'][str(path)],'frozen batch module source changed');alias='_fixed_adapter_batch_'+name;module=types.ModuleType(alias);module.__file__=str(path);sys.modules[alias]=module;sys.modules[name]=module;private.append(alias);exec(compile(raw,str(path),'exec'),module.__dict__);modules[name]=module
        need(modules['common'].EXPECTED_CONTRACT==c['batch_contract_sha256'],'mixed batch contract import');yield modules
    finally:
        for name,value in previous.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
        for alias in private:sys.modules.pop(alias,None)

def project(ordinal,batch_pin,tool_pin,result,parents,calls):
    """Pure selection after full verification; returns canonical immutable bytes."""
    need(type(ordinal)is int and ordinal in range(2,8),'only exact remaining ordinal2..7')
    need(result['status']=='all_six_three_saved_points_and_exact_owned_FD_calls_verified'and result['parent_roster']==list(range(2,8))and result['call_count']==42 and [r['ordinal']for r in result['parents']]==list(range(2,8))and result['outer_masses']==[.125]*8 and result['normalizers_posteriors']=='unverified','partial/wrong all-six proof cannot be projected')
    parent=parents[ordinal];need(parent['ordinal']==ordinal and [r['point_index']for r in parent['calls']]==[0,1,2],'selected parent base roster differs')
    names=[r['file']for r in parent['calls']]+[item['file']for row in parent['finite_differences']for item in row['calls']];need(len(names)==len(set(names))==7,'exact seven distinct selected calls required');selected={name:calls[name]for name in names}
    return canonical(dict(schema='VerifiedBatchParentWitnessV1',adapter_contract_sha256=EXPECTED_CONTRACT,ordinal=ordinal,batch_success=batch_pin,batch_tool_return=tool_pin,all_six_comparison=result,parent=parent,calls=selected,normalizers_posteriors='unverified',scope='selected immutable seven calls only after complete read-only42call batch verification; not fixed or normalized'))

def select_witness(c,ordinal,tool_path,tool_sha):
    need(type(ordinal)is int and ordinal in range(2,8),'independently selected ordinal required');source=Path(c['batch_source']);root=Path(c['batch_attempt_root']);out=root/'dynamic-six';tool_path=Path(tool_path);need(sha(tool_path)==tool_sha,'retained full batch tool-return changed')
    with frozen_batch(c)as batch:
        bc=batch['common'];bcontract=bc.read_contract();need(bc.sha(source/'contract.json')==c['batch_contract_sha256'],'batch contract changed');marker=batch['completion'].completed(out,tool_path,tool_sha)
        auth=json.loads((out/'authorization.json').read_text());review_path=Path(c['batch_admission_review']);need(pin(review_path)==c['batch_admission_review_pin']and auth['review']==json.loads(review_path.read_text())and auth['review_digest']==bc.digest(auth['review']),'batch was not completed under the independently selected review')
        comparison,_=batch['comparison_gate'].execute(root,bcontract,auth['review'],auth['review_digest'],read_only=True)
        # Entire comparison has passed. Only now read/select this parent's immutable receipts.
        result=json.loads((out/'result.json').read_text());parents={};calls={}
        for row in result['parents']:
            path=out/row['file'];need(bc.pin(path)==row['pin'],'batch parent index changed');parent=json.loads(path.read_text());parents[row['ordinal']]=parent
            if row['ordinal']==ordinal:
                for item in [*parent['calls'],*[item for entry in parent['finite_differences']for item in entry['calls']]]:
                    path=out/item['file'];need(bc.pin(path)==item['pin'],'selected call changed');calls[item['file']]=json.loads(path.read_bytes())
        data=project(ordinal,bc.pin(out/'controller-success.json'),bc.pin(tool_path),comparison,parents,calls)
        # Recheck full external acknowledgment at the publication boundary.
        batch['completion'].completed(out,tool_path,tool_sha)
    read_contract();return data
