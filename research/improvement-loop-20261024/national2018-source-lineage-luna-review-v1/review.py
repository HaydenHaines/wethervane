"""Independent read/hash-only recount of the retained national2018 source lineage."""
import hashlib
import json
from pathlib import Path

B = Path('/home/hayden/projects/wethervane-redesign-20261006/integration')
H = B / 'research/senate-stack-integration-20261020/national-2018-gpu-preparation-root-v1'
A = H / 'gpu-production-v1/attempt1'
P = Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/national2018-source-lineage-sol-v1')
SELECTED = 'research/senate-stack-integration-20261020/national-problem-assembly-astra-v1/'
OMITTED = {
    'assemble.py': 'e28e5f715051f4f3aab0323e403250e57d1334535415d3348312164ef1d4888b',
    'dependencies.py': 'f36a835dc82fc02e8963be81eb27db6d0d538aab19d68b150e0dd7039f44b95f',
    'inputs_v3.py': '85adb960b982be3e5dd10e2cbd29521a209ad380e22e323ad97646c523a96180',
    'reader.py': '70f9fda6564f5890d8bc9c80bb782d6e3d0762b4b454e2e53d010fe3db887b95',
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def pin(path):
    return {'sha256': sha(path), 'bytes': Path(path).stat().st_size}

def main():
    plan_path = A / 'plan.json'
    plan = load(plan_path)
    launch = load(A / 'launch.json')
    receipt = load(A / 'receipt.json')
    sampling = load(A / 'sampling-receipt.json')
    original = load(B / plan['numerical_preflight']['original_plan']['path'])
    rows = []
    for i, (rel, expected) in enumerate(plan['source_pins'].items()):
        snap = A / 'source-snapshots' / f'{i:03d}-{Path(rel).name}'
        current = B / rel
        rows.append({
            'path': rel, 'expected': expected,
            'snapshot_sha256': sha(snap), 'snapshot_matches': sha(snap) == expected,
            'current_sha256': sha(current) if current.is_file() else None,
            'current_matches': current.is_file() and sha(current) == expected,
            'original_scientific_plan_matches': original['source_pins'].get(rel) == expected,
            'snapshot_bytes': snap.stat().st_size,
        })
    selected = [r for r in rows if r['path'].startswith(SELECTED)]
    selected_names = {Path(r['path']).name for r in selected}
    omitted = {}
    for name, expected in OMITTED.items():
        hit = [r for r in selected if Path(r['path']).name == name]
        assert len(hit) == 1 and hit[0]['expected'] == expected
        selected_index = next(i for i, (rel, _) in enumerate(plan['source_pins'].items())
                              if rel.startswith(SELECTED) and Path(rel).name == name)
        retained = P / 'selected-snapshots' / f'{selected_index:03d}-{name}'
        omitted[name] = {
            'expected': expected,
            'plan_snapshot_current_all_match': hit[0]['snapshot_matches'] and hit[0]['current_matches'],
            'old_scientific_plan_matches': hit[0]['original_scientific_plan_matches'],
            'retained_copy_sha256': sha(retained),
            'retained_copy_matches_expected': sha(retained) == expected,
        }
    assert len(rows) == 165 and len(list((A / 'source-snapshots').iterdir())) == 165
    assert len(selected) == 11 and len(selected_names) == 11
    assert all(r['snapshot_matches'] and r['current_matches'] for r in rows)
    assert all(r['original_scientific_plan_matches'] for r in selected)
    assert pin(plan_path)['sha256'] == pin(H / 'gpu-production-v1/plan.json')['sha256']
    assert pin(plan_path)['sha256'] == launch['plan']['sha256'] == sampling['plan']['sha256']
    runtime = B / plan['runtime']['path']
    assert sha(runtime) == plan['runtime']['sha256']
    assert load(runtime)['inventory'] == plan['runtime_inventory']
    for rel, metadata in plan['source_inputs'].items():
        assert sha(A / 'input-snapshots' / f'{rel}.json') == metadata['sha256']
    return {
        'schema': 'National2018SourceLineageLunaIndependentReviewV1',
        'plan_source_count': len(rows),
        'snapshot_match_count': sum(r['snapshot_matches'] for r in rows),
        'current_match_count': sum(r['current_matches'] for r in rows),
        'snapshot_bytes_read': sum(r['snapshot_bytes'] for r in rows),
        'selected_package_count': len(selected),
        'selected_package_all_original_plan_snapshot_current_match': True,
        'four_previously_omitted_files': omitted,
        'plan_hash': pin(plan_path)['sha256'],
        'launch_receipt_sampling_plan_hashes_equal': True,
        'runtime_inventory_matches': True,
        'input_snapshots_match_plan': True,
        'receipt_status': receipt.get('status'),
        'boundary': 'Read/hash-only; no model import, graph construction, NPZ or sampler-array read; local hashes do not prove bytes interpreted at import time or immutable historical execution.'
    }

if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True, indent=2))
