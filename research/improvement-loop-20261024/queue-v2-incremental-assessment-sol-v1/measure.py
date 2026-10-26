"""Bounded real V2 operation counts/timings, temporary synthetic fixtures only."""
import collections
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;LAB=HERE.parent
sys.path.insert(0,str(LAB/'queue-v2'))
import experiment_queue as q
import execution as ex
import test_experiment_queue as fixtures


def fixture(base):
    base.mkdir();f=fixtures.QueueTests('runTest');f.base=base;f.root=base/'queue';f.source=base/'owner.py'
    f.source.write_text(fixtures.FIXTURE);q.initialize(f.root,1);return f


def measured(f,action):
    counts=collections.Counter();pin,checked,entries,iterdir=ex.pin,ex.checked,q.entries,Path.iterdir
    sources={str((LAB/'queue-v2'/name).resolve()) for name in ('experiment_queue.py','execution.py','ledger.py','run_receipts.py')}
    def observed_pin(path):
        item=pin(path);counts['pin_calls']+=1;counts['pinned_bytes']+=item['size']
        if item['path'] in sources:counts['source_pin_calls']+=1;counts['source_pinned_bytes']+=item['size']
        return item
    def observed_checked(item):counts['checked_calls']+=1;counts['checked_bytes']+=item['size'];return checked(item)
    def observed_entries(*args,**kwargs):counts['history_scans']+=1;return entries(*args,**kwargs)
    def observed_iterdir(path):
        if path==f.root/'queues':
            counts['queue_directory_listings']+=1
            for item in iterdir(path):counts['queue_directory_entries']+=1;yield item
        else:yield from iterdir(path)
    start=time.monotonic()
    with patch.object(ex,'pin',observed_pin),patch.object(ex,'checked',observed_checked),patch.object(q,'entries',observed_entries),patch.object(Path,'iterdir',observed_iterdir):value=action()
    return value,dict(seconds=time.monotonic()-start,**counts)


def admit(f,i):
    name='p'+str(i);plan=dict(schema='LocalExperimentQueuePlanV1',queue_id=name,jobs=[f.job(name)])
    return q.enqueue(f.root,plan,q.digest(plan))


def bytes_used(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def main():
    start=time.monotonic();cpu=time.process_time();usage_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with tempfile.TemporaryDirectory() as temporary:
        base=Path(temporary);pending=fixture(base/'pending');completed=fixture(base/'completed')
        result=dict(schema='QueueV2BoundedScaleAssessmentV1',pending=[],completed=[],limits=dict(pending_admissions=64,real_commands=8,capacity=1,nice=15),qualification='unassessed')
        try:
            for i in range(64):
                _,metric=measured(pending,lambda i=i:admit(pending,i));result['pending'].append(dict(existing_plans=i,measurement=metric))
            _,result['pending_status']=measured(pending,lambda:q.status(pending.root))
            for i in range(8):
                _,admission=measured(completed,lambda i=i:admit(completed,i));q.tick(completed.root)
                deadline=time.monotonic()+8
                while time.monotonic()<deadline:
                    state=q.tick(completed.root)
                    if state['all_jobs_recorded']:break
                    time.sleep(.02)
                assert state['all_jobs_recorded'] and state['reserved']==0
                _,idle=measured(completed,lambda:q.tick(completed.root));result['completed'].append(dict(history=i+1,admission=admission,idle_tick=idle))
            result['temp_file_bytes_at_end']=bytes_used(base)
            result['temp_file_count_at_end']=sum(p.is_file() for p in base.rglob('*'))
            assert not (pending.root/'executions').exists()
        finally:pending.cleanup();completed.cleanup()
    usage_after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result['wall_seconds']=time.monotonic()-start;result['controller_cpu_seconds']=time.process_time()-cpu
    result['reaped_child_cpu_seconds']=(usage_after.ru_utime+usage_after.ru_stime)-(usage_before.ru_utime+usage_before.ru_stime)
    result['controller_max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['temporary_stores_removed']=not base.exists()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
