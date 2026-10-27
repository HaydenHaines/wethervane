"""Use-time reviewed snapshot comparator, preserved trusted-local boundary."""
import hashlib,sys,types
from pathlib import Path
import common
from common import *
def execute(root,c,review,review_sha,read_only=False):
 snapshot=reviewed_sources(review,review_sha,EXPECTED_CONTRACT);read_contract();need(Path(common.__file__).resolve()==(HERE/'common.py').resolve(),'cached common path differs')
 import numpy as np
 expected=c['runtime_expected']['numpy'];need(np.__version__==expected['version']and Path(np.__file__).resolve()==Path(expected['path']).resolve(),'NumPy runtime differs')
 names=['evaluator','selection','baseline_compare'];previous={n:sys.modules.get(n)for n in names}
 try:
  for name in names:
   module=types.ModuleType(name);module.__file__=str(HERE/(name+'.py'));sys.modules[name]=module;exec(compile(snapshot[name+'.py'],module.__file__,'exec'),module.__dict__)
  module=types.ModuleType('_verified_all_six');module.__file__=str(HERE/'compare.py');exec(compile(snapshot['compare.py'],module.__file__,'exec'),module.__dict__)
  if read_only:
   def verify_existing(path,value):need(canonical(__import__('json').loads(Path(path).read_text()))==canonical(value),'read-only batch comparison drift')
   module.atomic_json=verify_existing
  result=module.compare_batch(root,c)
 finally:
  for name,old in previous.items():
   if old is None:sys.modules.pop(name,None)
   else:sys.modules[name]=old
 reviewed_sources(review,review_sha,EXPECTED_CONTRACT);read_contract();return result,dict(schema='SixParentComparisonExecutionV1',phase='dynamic-six',parent_roster=list(range(2,8)),contract_sha256=EXPECTED_CONTRACT,review_digest=review_sha,source_sha256={n:hashlib.sha256(v).hexdigest()for n,v in snapshot.items()},scope='verified snapshots in trusted local workspace')
