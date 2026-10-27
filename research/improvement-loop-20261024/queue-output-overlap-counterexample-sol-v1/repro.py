"""Admission-only counterexample; creates no execution child or output directory."""
import json,sys
from pathlib import Path
source=Path(sys.argv[1]).resolve();sys.path.insert(0,str(source))
import experiment_queue as q
from test_experiment_queue import QueueTests
fx=QueueTests(methodName='runTest');fx.setUp()
try:
 def job(name,path):
  j=fx.job(name);j['execution_spec']['output_directory']=str(path);j['execution_spec']['argv'][2]=str(path);return j
 def plan(name,jobs):return dict(schema='LocalExperimentQueuePlanV1',queue_id=name,jobs=jobs)
 p=plan('one',[job('a',fx.base/'owned'),job('b',fx.base/'owned-other'),job('c',fx.base/'owned'/'child')]);q.enqueue(fx.root,p,q.digest(p))
 p=plan('two',[job('d',fx.base/'owned'/'another-child')]);q.enqueue(fx.root,p,q.digest(p))
 print(json.dumps({'source':str(source),'same_plan_three_paths':'accepted','cross_plan_nested_path':'accepted','execution_store_exists':(fx.root/'executions').exists(),'output_directories_exist':[(fx.base/'owned').exists(),(fx.base/'owned-other').exists()]}))
finally:fx.doCleanups()
