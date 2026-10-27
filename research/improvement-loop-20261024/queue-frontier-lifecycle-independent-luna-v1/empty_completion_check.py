"""Check machine-readable completion fields for a fresh empty research store."""
import json
import tempfile
from pathlib import Path
import sys

SOURCE = Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/queue-frontier-lifecycle-sol-v1')
sys.path.insert(0, str(SOURCE))
import dispatch

with tempfile.TemporaryDirectory(prefix='luna-empty-lifecycle-') as temp:
    root = dispatch.initialize(Path(temp) / 'empty')
    with dispatch.opened(root / 'admissions.sqlite') as db:
        result = dispatch.completion(db)
print(json.dumps(result, sort_keys=True, indent=2))
assert result['state'] == 'empty'
assert result['total'] == 0
assert result['all_terminal'] is True
assert result['all_executions_accepted_unqualified'] is False
