"""Preserved v1 bug: accepted nested outputs separated in lexical ordering."""
import json
from pathlib import Path
import tempfile
import core
import test_experiment_queue as fixtures

with tempfile.TemporaryDirectory() as directory:
    base=Path(directory)
    f=fixtures.QueueTests('runTest'); f.base=base; f.source=base/'owner.py'
    f.source.write_text(fixtures.FIXTURE)
    jobs=[f.job('job'+str(i)) for i in range(3)]
    for job, suffix in zip(jobs,['owned','owned-other','owned/child']):
        job['execution_spec']['output_directory']=str(base/suffix)
    plan=dict(schema='LocalExperimentQueuePlanV1',queue_id='nested-bug',jobs=jobs)
    db=base/'admissions.sqlite';core.initialize(db)
    core.admit(db,plan,core.queue.digest(plan))
    assert core.operational_status(db)['ownership_claims']==3
    assert core.audit(db)['structural_audit']=='passed'
    print(json.dumps(dict(v1_nested_claims_incorrectly_accepted=True,v1_structural_audit_incorrectly_passed=True,executions=0)))
