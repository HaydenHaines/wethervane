"""Bounded real queue probe; synthetic commands only; retain isolated stores."""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import experiment_queue as q
import execution as ex
import test_experiment_queue as fixtures


def fixture(name):
    f = fixtures.QueueTests('runTest')
    f.base = HERE / name
    f.base.mkdir()
    f.root = f.base / 'queue'
    f.source = f.base / 'owner.py'
    f.source.write_text(fixtures.FIXTURE)
    q.initialize(f.root, 1)
    return f


def timed(action):
    start = time.monotonic()
    with patch.object(q, 'entries', wraps=q.entries) as scans, patch.object(ex, 'pin', wraps=ex.pin) as pins:
        value = action()
    return value, dict(seconds=time.monotonic()-start, historyScans=scans.call_count, pinCalls=pins.call_count)


def admit(f, n):
    name = 'cycle-' + str(n)
    plan = dict(schema='LocalExperimentQueuePlanV1', queue_id=name, jobs=[f.job(name)])
    return q.enqueue(f.root, plan, q.digest(plan))


def main():
    result = dict(schema='AdaptiveQueueBoundedScaleProbeV1', qualification='unassessed', pending=[], incremental=[])
    pending = fixture('pending-store')
    for i in range(160):
        _, measurement = timed(lambda: admit(pending, i))
        result['pending'].append(dict(existingPlans=i, enqueue=measurement))
    for n in (1, 40, 80, 160):
        # All prior admissions remain present: status below measures full 160 store.
        pass
    state, metric = timed(lambda: q.status(pending.root))
    assert len(state['queues']) == 160 and state['reserved'] == 0
    assert not (pending.root / 'executions').exists()
    result['pendingStatus160'] = metric
    result['pendingIntegrity'] = dict(plans=160, reservations=0, launchedAttempts=0)
    active = fixture('incremental-store')
    deadline = time.monotonic()+45
    for i in range(16):
        _, admission = timed(lambda: admit(active, i))
        _, dispatch = timed(lambda: q.tick(active.root))
        polls = 0
        while True:
            assert time.monotonic() < deadline, 'bounded runtime exceeded'
            time.sleep(.03)
            state = q.tick(active.root)
            polls += 1
            if state['all_jobs_recorded']:
                break
        _, idle = timed(lambda: q.tick(active.root))
        result['incremental'].append(dict(completedPlans=i+1, enqueue=admission, dispatch=dispatch, idleTick=idle, reconciliationPolls=polls))
    clean = q.status(active.root)
    assert clean['all_jobs_recorded'] and clean['reserved'] == 0
    assert all(j['outcome']=='accepted_execution' for jobs in clean['queues'].values() for j in jobs.values())
    # Detect historical receipt drift without allowing dispatch to proceed.
    target = active.root/'queues'/'cycle-0'/'plan.json'
    original = target.read_bytes()
    try:
        target.write_bytes(original+b' ')
        bad = q.tick(active.root)
        assert bad['quarantined_admissions'] and not bad['all_jobs_recorded']
    finally:
        target.write_bytes(original)
    assert q.status(active.root)['all_jobs_recorded']
    result['incrementalIntegrity'] = dict(completedSyntheticCommands=16, reservations=0, historicalPlanTamperDetected=True, restoredOriginalBytes=True)
    result['sources'] = [ex.pin(HERE/'probe.py'), ex.pin(HERE.parent/'experiment_queue.py'), ex.pin(HERE.parent/'execution.py'), ex.pin(HERE.parent/'test_experiment_queue.py')]
    (HERE/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pending','incremental','sources')}))


if __name__ == '__main__':
    main()
