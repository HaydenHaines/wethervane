"""Bounded actual queue fault/restart probe. Tiny commands, temporary domains only."""
import copy
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
LAB=HERE.parent
sys.path.insert(0,str(LAB))
import experiment_queue as original_q
import execution as ex
import test_experiment_queue as fixtures


def killed_admission(module_directory,root,plan,boundary_name="plan.json"):
    sys.path.insert(0,module_directory)
    # Distinct process imports the selected real implementation.
    sys.modules.pop('experiment_queue',None)
    queue=importlib.import_module('experiment_queue')
    publish=queue.executor.publish
    def boundary(path,value):
        result=publish(path,value)
        if Path(path)==Path(root)/'queues'/plan['queue_id']/boundary_name:os._exit(82)
        return result
    queue.executor.publish=boundary
    queue.enqueue(root,plan,queue.digest(plan))


def domain(base,queue):
    f=fixtures.QueueTests('runTest');f.base=base;f.root=base/'queue';f.source=base/'owner.py'
    f.source.write_text(fixtures.FIXTURE);queue.initialize(f.root,1)
    return f


def complete(queue,root,deadline_seconds=8):
    deadline=time.monotonic()+deadline_seconds
    while time.monotonic()<deadline:
        state=queue.tick(root)
        if state['reserved']==0 and all(j['outcome']=='accepted_execution' for jobs in state['queues'].values() for j in jobs.values()):return state
        time.sleep(.02)
    raise AssertionError(state)


def partial_probe(queue,directory):
    with tempfile.TemporaryDirectory() as temporary:
        f=domain(Path(temporary),queue)
        try:
            p=dict(schema='LocalExperimentQueuePlanV1',queue_id='original',jobs=[f.job('same')])
            child=mp.Process(target=killed_admission,args=(str(directory),str(f.root),p));child.start();child.join(8)
            assert child.exitcode==82
            assert (f.root/'queues/original/plan.json').exists() and not (f.root/'queues/original/receipt.json').exists()
            contender=copy.deepcopy(p);contender['queue_id']='contender'
            admitted=False;contender_error=None
            try:queue.enqueue(f.root,contender,queue.digest(contender));admitted=True
            except ValueError as error:contender_error=str(error)
            repair_error=None
            try:queue.enqueue(f.root,p,queue.digest(p))
            except ValueError as error:repair_error=str(error)
            state=complete(queue,f.root)
            launches=len(list((f.root/'executions/attempts').glob('*/launch-intent.json')))
            return dict(contender_admitted=admitted,contender_error=contender_error,original_exact_repair_error=repair_error,
                partial_admissions=state['partial_admissions'],all_jobs_recorded=state['all_jobs_recorded'],reserved=state['reserved'],
                queues=state['queues'],actual_launches=launches,caller_exitcode=child.exitcode)
        finally:f.cleanup()


def main():
    result=dict(schema='OriginalQueuePartialOwnershipFaultProbeV1',original=partial_probe(original_q,LAB))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
