"""Isolated tiny child then stale-comparator challenge, no model."""
import json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);mode=sys.argv[2];sys.path.insert(0,str(root));import common as c,comparison_gate as gate
(root/'compare.py').write_text("from pathlib import Path\ndef compare_batch(root,c):\n Path(root,'compared').write_text('verified')\n return {'fixture':True}\n");(root/'evaluator.py').write_text('# invented evaluator fixture only\n')
names=['common.py','child.py','launch.py','compare.py','test_tiny.py','contract.json','entry.py','evaluator.py','comparison_gate.py','baseline_compare.py','selection.py','completion.py'];review={'schema':'OwnedParentEvaluatorIndependentReviewV1','decision':'allow_one_bounded_attempt','allowed_phases':['dynamic-six'],'parent_roster':list(range(2,8)),'contract_sha256':c.EXPECTED_CONTRACT,'packet_pins':{n:c.sha(root/n)for n in names}};retained=c.digest(review);child=subprocess.run([sys.executable,'-c','print("tiny complete")'],capture_output=True,timeout=5);assert child.returncode==0
if mode=='drift':(root/'compare.py').write_text("from pathlib import Path\nPath(__file__).with_name('bad-import').write_text('bad')\ndef compare_batch(root,c):return {}\n")
rejected=False;error=None
try:value,execution=gate.execute(root,c.read_contract(),review,retained);c.atomic_json(root/'controller-success.json',{'fixture':value})
except ValueError as exc:rejected=True;error=str(exc)
print(json.dumps({'mode':mode,'child_returncode':child.returncode,'rejected':rejected,'error':error,'compared':(root/'compared').exists(),'bad_import':(root/'bad-import').exists(),'success':(root/'controller-success.json').exists()}))
