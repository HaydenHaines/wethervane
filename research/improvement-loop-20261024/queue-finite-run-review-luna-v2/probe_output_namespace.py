"""Independent admission-only challenge of V2 output namespace ownership."""
import json, os, subprocess, sys
from pathlib import Path
source=Path(sys.argv[1]).resolve();sys.path.insert(0,str(source))
import experiment_queue as q
from test_experiment_queue import QueueTests
fx=QueueTests(methodName="runTest");fx.setUp()
def make(name,path):
 j=fx.job(name);j["execution_spec"]["output_directory"]=str(path);j["execution_spec"]["argv"][2]=str(path)
 return {"schema":"LocalExperimentQueuePlanV1","queue_id":"q-"+name,"jobs":[j]}
def accept(plan):q.enqueue(fx.root,plan,q.digest(plan))
def reject(plan):
 try:accept(plan)
 except Exception:return True
 return False
try:
 root=fx.base/"same-plan";fx.root=root;q.initialize(root,2)
 outer=fx.base/"owned";sibling=fx.base/"owned-other";child=outer/"child"
 jobs=[make("outer",outer)["jobs"][0],make("sibling",sibling)["jobs"][0],make("child",child)["jobs"][0]]
 same={"schema":"LocalExperimentQueuePlanV1","queue_id":"same","jobs":jobs}
 print("same_plan_outer_sibling_child_rejected",reject(same))
 root=fx.base/"accepted";fx.root=root;q.initialize(root,2);accept(make("base",outer))
 print("accepted_claim_rejects_descendant",reject(make("desc",child)))
 print("accepted_claim_rejects_ancestor",reject(make("anc",fx.base)))
 print("accepted_claim_rejects_exact",reject(make("exact",outer)))
 print("accepted_claim_allows_lexical_sibling",not reject(make("sibling",sibling)))
 root=fx.base/"reverse";fx.root=root;q.initialize(root,2);accept(make("descfirst",child))
 print("reverse_order_rejects_ancestor",reject(make("ancestor",outer)))
 root=fx.base/"pending";fx.root=root;q.initialize(root,2);pending=make("pending",outer)
 code="import json,sys;from pathlib import Path;import experiment_queue as q;r=Path(sys.argv[1]);p=json.loads(sys.argv[2]);q.enqueue(r,p,q.digest(p))"
 env=dict(os.environ,PYTHONPATH=str(source),FINITE_QUEUE_CRASH="intent_after_commit",PYTHONDONTWRITEBYTECODE="1")
 childproc=subprocess.run([sys.executable,"-B","-c",code,str(root),json.dumps(pending)],env=env,capture_output=True);assert childproc.returncode==91
 print("pending_claim_rejects_descendant",reject(make("pending-desc",child)))
 print("pending_claim_rejects_file_directory_alias",reject(make("pending-file",outer/"data.txt")))
 print("pending_claim_allows_sibling",not reject(make("pending-sibling",sibling)))
 print("no_output_roots_materialized",not outer.exists() and not child.exists() and not sibling.exists())
finally:fx.doCleanups()
