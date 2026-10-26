"""Isolated invented child-then-comparator fixture; no scientific graph."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();mode=sys.argv[2];sys.path.insert(0,str(root))
import common as c,comparison_gate as gate
fixture="from pathlib import Path\ndef compare_dynamic(root,contract):\n Path(root,'compared').write_text('verified snapshot executed')\n return {'fixture':True}\n"
(root/'compare.py').write_text(fixture);(root/'evaluator.py').write_text('# invented empty evaluator module; no model\n')
review={'schema':'OwnedParentEvaluatorIndependentReviewV1','decision':'allow_one_bounded_attempt','allowed_phases':['dynamic'],'contract_sha256':c.EXPECTED_CONTRACT,'packet_pins':{n:c.sha(root/n)for n in ['common.py','entry.py','child.py','evaluator.py','launch.py','compare.py','test_tiny.py','contract.json','comparison_gate.py']}};retained=c.digest(review)
child=subprocess.run([sys.executable,'-c','print("invented child complete")'],capture_output=True,timeout=5);assert child.returncode==0
if mode=='drift':(root/'compare.py').write_text("from pathlib import Path\nPath(__file__).with_name('unreviewed-executed').write_text('BAD')\ndef compare_dynamic(root,contract): return {}\n")
rejected=False;error=None;execution=None
try:
 value,execution=gate.execute(root,c.read_contract(),review,retained,'dynamic');c.atomic_json(root/'controller-success.json',{'fixture':value,'execution':execution})
except ValueError as exc:rejected=True;error=str(exc)
result={'mode':mode,'child_returncode':child.returncode,'rejected':rejected,'error':error,'compared':(root/'compared').exists(),'unreviewed_executed':(root/'unreviewed-executed').exists(),'success_published':(root/'controller-success.json').exists(),'execution':execution};print(json.dumps(result))
