"""Inspect optimizer plans in one fresh empty research store; no jobs executed."""
import json
from pathlib import Path
import tempfile
import dispatch as d

QUERIES={
    'active_reservations': 'SELECT r.attempt FROM reservations r LEFT JOIN releases l ON l.attempt=r.attempt WHERE l.attempt IS NULL',
    'pending_candidates': 'SELECT o.attempt FROM owners o LEFT JOIN reservations r ON r.attempt=o.attempt WHERE r.attempt IS NULL ORDER BY o.rowid',
}
with tempfile.TemporaryDirectory() as temporary:
    root=d.initialize(Path(temporary)/'queue')
    with d.admission.opened(root/'admissions.sqlite') as db:
        result={name:dict(sql=query,optimizer_rows=db.execute('EXPLAIN QUERY PLAN '+query).fetchall()) for name,query in QUERIES.items()}
    print(json.dumps(dict(schema='IncrementalDispatchSQLPlanInspectionV1',empty_fresh_store=True,jobs_executed=0,queries=result,source_pins=d.current_sources()),indent=2))
