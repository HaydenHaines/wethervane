"""Review-local verified source snapshots, including selected-parent/base comparator."""
import hashlib,sys,types
from pathlib import Path
import common
from common import *
def execute(root,c,review,review_sha,phase,ordinal,read_only=False):
 snapshot=common.reviewed_sources(review,review_sha,EXPECTED_CONTRACT);common.read_contract();need(Path(common.__file__).resolve()==(HERE/'common.py').resolve(),'cached common path differs')
 import numpy as np
 expected=c['runtime_expected']['numpy'];need(np.__version__==expected['version']and Path(np.__file__).resolve()==Path(expected['path']).resolve(),'NumPy runtime differs')
 names=['evaluator','batch_adapter','selection','baseline_compare'];previous={n:sys.modules.get(n)for n in names}
 try:
  for name in names:
   m=types.ModuleType(name);m.__file__=str(HERE/(name+'.py'));sys.modules[name]=m;exec(compile(snapshot[name+'.py'],m.__file__,'exec'),m.__dict__)
  m=types.ModuleType('_remaining_verified_comparator');m.__file__=str(HERE/'compare.py');exec(compile(snapshot['compare.py'],m.__file__,'exec'),m.__dict__)
  if read_only:
   def verify_existing(path,value):need(canonical(__import__('json').loads(Path(path).read_text()))==canonical(value),'read-only replay comparison drift')
   m.atomic_json=verify_existing
  need(phase=='fixed-from-batch','only fixed-from-batch comparator');value=m.compare_fixed_from_batch(root,c,ordinal)
 finally:
  for name,old in previous.items():
   if old is None:sys.modules.pop(name,None)
   else:sys.modules[name]=old
 common.reviewed_sources(review,review_sha,EXPECTED_CONTRACT);common.read_contract();return value,{'schema':'RemainingVerifiedComparisonExecutionV1','selected_parent_ordinal':ordinal,'phase':phase,'source_sha256':{n:hashlib.sha256(v).hexdigest()for n,v in snapshot.items()},'review_digest':review_sha,'contract_sha256':EXPECTED_CONTRACT,'scope':'trusted-local verified snapshots, not arbitrary hostile filesystem security'}
