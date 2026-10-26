"""Bounded restart/concurrency audit of unchanged existing queue, no patch import."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import probe


def main():
    queue=probe.original_q;start=time.monotonic()
    with tempfile.TemporaryDirectory() as temporary:
        f=probe.domain(Path(temporary),queue)
        try:
            observations=[]
            for i in range(12):
                name='cycle'+str(i);job=f.job(name)
                if i:
                    previous=f.root/('queues/cycle'+str(i-1)+' /jobs/cycle'+str(i-1)+'/result.json').replace(' ','')
                    assert queue.document(previous)['outcome']=='accepted_execution'
                    job['execution_spec']['input_pins'].append(queue.executor.pin(previous))
                plan=dict(schema='LocalExperimentQueuePlanV1',queue_id=name,jobs=[job])
                saved=queue.enqueue(f.root,plan,queue.digest(plan))
                assert queue.enqueue(f.root,plan,queue.digest(plan))==saved
                code='import sys,os;sys.path.insert(0,sys.argv[1]);import experiment_queue as q;q.tick(sys.argv[2]);os._exit(83)'
                clients=[subprocess.Popen([sys.executable,'-B','-c',code,str(probe.LAB),str(f.root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(2)]
                for client in clients:
                    out,error=client.communicate(timeout=8);assert client.returncode==83,(out,error)
                state=probe.complete(queue,f.root);assert state['all_jobs_recorded'] and not state['reserved']
                observations.append(dict(cycle=i+1,recorded_jobs=sum(len(jobs) for jobs in state['queues'].values()),reserved=state['reserved']))
            launches=len(list((f.root/'executions/attempts').glob('*/launch-intent.json')))
            releases=len(list(f.root.glob('queues/*/jobs/*/release.json')))
            assert launches==releases==12
            print(json.dumps(dict(schema='ExistingQueueBoundedSerialRestartV1',source=str(Path(queue.__file__).resolve()),cycles=12,competing_tick_clients=24,clients_exited_after_tick=True,actual_launches=launches,releases=releases,observations=observations,seconds=time.monotonic()-start,qualification='unassessed'),indent=2))
        finally:f.cleanup()


if __name__=='__main__':main()
