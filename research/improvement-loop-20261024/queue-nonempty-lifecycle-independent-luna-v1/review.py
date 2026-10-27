"""Read-only source-delta/pin and consumer-boundary challenge."""
import ast
import hashlib
import json
from pathlib import Path
import sys

BASE = Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024')
OLD = BASE / 'queue-frontier-lifecycle-sol-v1'
NEW = BASE / 'queue-nonempty-lifecycle-sol-v1'
CHANGED = {'initialize', 'opened', 'locked', 'completion'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def functions(path):
    tree = ast.parse(Path(path).read_text())
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

def ast_sha(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()

def main():
    pins = json.loads((NEW / 'source-pins.json').read_text())['pins']
    bad_pins = [p['path'] for p in pins if not Path(p['path']).is_file() or sha(p['path']) != p['sha256']]
    oldf, newf = functions(OLD / 'dispatch.py'), functions(NEW / 'dispatch.py')
    changed = sorted(n for n in set(oldf) | set(newf)
                     if n not in oldf or n not in newf or ast.dump(oldf[n], include_attributes=False) != ast.dump(newf[n], include_attributes=False))
    delta = json.loads((NEW / 'source-delta.json').read_text())
    unchanged = sorted(set(oldf) & set(newf) - CHANGED)
    ast_mismatches = [n for n in unchanged if ast_sha(oldf[n]) != ast_sha(newf[n]) or ast_sha(newf[n]) != delta['unchanged_function_ast_sha256'].get(n)]
    sys.path.insert(0, str(NEW))
    import consumer
    fabricated = {
        'schema': 'NonemptyCurrentAdmissionsLifecycleV1',
        'state': 'nonempty_current_admissions_terminal_all_accepted_unqualified',
        'total': 1, 'pending': 0, 'active': 0, 'executed_terminal': 1,
        'blocked_terminal': 0, 'accepted_execution': 1, 'failed_execution': 0,
        'nonempty_current_admissions_all_terminal': True,
        'nonempty_current_admissions_all_accepted_unqualified': True
    }
    fabricated_is_accepted = consumer.accepted_unqualified_work(fabricated)
    assert not bad_pins and changed == sorted(CHANGED) and not ast_mismatches
    assert fabricated_is_accepted is True
    return {
        'schema': 'QueueNonemptyLifecycleIndependentSourceReviewLunaV1',
        'source_pin_count': len(pins), 'source_pin_mismatches': bad_pins,
        'changed_dispatch_functions': changed,
        'unchanged_dispatch_functions_ast_match_count': len(unchanged) - len(ast_mismatches),
        'unchanged_dispatch_functions_ast_mismatches': ast_mismatches,
        'consumer_self_consistent_fabricated_success_accepted': fabricated_is_accepted,
        'consumer_limit': 'The helper is a shape/state consistency fixture, not a cryptographic/authenticated receipt verifier; its input must come from the trusted dispatcher snapshot.'
    }

if __name__ == '__main__':
    print(json.dumps(main(), indent=2, sort_keys=True))
