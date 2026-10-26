"""Research-only supplied-claim checks; no authentication or fit admission."""
from collections.abc import Mapping
from datetime import datetime
from itertools import combinations


def validate(ledger):
    errors = []

    def need(ok, message):
        if not ok:
            errors.append(message)
        return ok

    def result():
        return dict(status='contract_checks_passed' if not errors else 'unresolved_or_rejected',
                    errors=errors, authenticated=False, fit_authorized=False,
                    qualification='Checks supplied claims only; source authentication, completeness and scientific review remain separate.')

    def text(value):
        return isinstance(value, str) and bool(value.strip())

    def strings(value, field, *, nonempty=False):
        if not need(isinstance(value, list), field + ': list required'):
            return []
        valid = all(text(x) for x in value)
        need(valid, field + ': nonempty string elements required')
        if nonempty:
            need(bool(value), field + ': nonempty list required')
        # Never hash malformed input (dict/list elements may be unhashable).
        clean = [x for x in value if text(x)]
        need(len(set(clean)) == len(clean), field + ': duplicate elements')
        return clean

    def pinned(value):
        return (isinstance(value, Mapping) and text(value.get('path'))
                and isinstance(value.get('sha256'), str) and len(value['sha256']) == 64
                and all(c in '0123456789abcdef' for c in value['sha256']))

    def instant(value):
        if not isinstance(value, str):
            return None
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt if dt.tzinfo is not None and dt.utcoffset() is not None else None
        except (ValueError, TypeError, OverflowError):
            return None

    if not need(isinstance(ledger, Mapping), 'ledger: mapping required'):
        return result()
    mode = ledger.get('mode')
    need(mode in ('historical_replay', 'retrospective_joint'), 'unsupported mode')
    cutoff = instant(ledger.get('cutoff'))
    need(cutoff is not None, 'timezone-aware cutoff required')
    raw_records = ledger.get('records')
    if not need(isinstance(raw_records, list), 'records: list required'):
        raw_records = []
    need(bool(raw_records), 'records: nonempty list required')
    records = []
    ids = []
    for index, record in enumerate(raw_records):
        if not need(isinstance(record, Mapping), f'record {index}: mapping required'):
            continue
        rid = record.get('id')
        if not need(text(rid), f'record {index}: nonempty string ID required'):
            continue
        records.append(record)
        ids.append(rid)
    need(len(set(ids)) == len(ids), 'unique record IDs required')
    atoms = {}
    labels_used = set()
    label_owners = {}
    targets = set(strings(ledger.get('evaluation_event_keys'), 'evaluation_event_keys'))
    for record in records:
        rid = record['id']
        need(pinned(record.get('source')), f'{rid}: pinned source required')
        need(text(record.get('factor_owner')), f'{rid}: numerical factor owner required')
        own_atoms = strings(record.get('canonical_atoms'), f'{rid}: canonical_atoms', nonempty=True)
        need(record.get('canonical_atoms_status') == 'established' and bool(own_atoms), f'{rid}: canonical atoms unresolved')
        for atom in own_atoms:
            need(atom not in atoms, f'duplicate evidence atom {atom}')
            atoms[atom] = rid
        labels = strings(record.get('outcome_label_event_keys'), f'{rid}: outcome_label_event_keys')
        for label in set(labels):
            need(label not in label_owners, f'outcome label reused across factors {label}')
            label_owners[label] = rid
        labels_used.update(labels)
        if mode == 'historical_replay':
            available = instant(record.get('verified_available_at'))
            need(record.get('availability_status') == 'verified' and available is not None,
                 f'{rid}: historical availability unresolved; field end is not publication')
            if available is not None and cutoff is not None:
                need(available <= cutoff, f'{rid}: available after cutoff')
            need(record.get('historical_version_status') == 'verified', f'{rid}: historical version unresolved')
    need(not (labels_used & targets), 'evaluation outcome used as training input')
    edges = ledger.get('dependencies')
    if not need(isinstance(edges, list), 'dependencies: list required'):
        edges = []
    indexed = {}
    for index, edge in enumerate(edges):
        if not need(isinstance(edge, Mapping), f'dependency {index}: mapping required'):
            continue
        pair_values = strings(edge.get('records'), f'dependency {index}: records', nonempty=True)
        pair = frozenset(pair_values)
        if not need(len(pair_values) == 2 and len(pair) == 2 and pair <= set(ids),
                    f'dependency {index}: exactly two known distinct record IDs required'):
            continue
        need(pair not in indexed, 'duplicate dependency assessment')
        indexed[pair] = edge
        # Validate every edge, not only those reached by a later pair lookup.
        for axis in ('respondents', 'source_error'):
            claim = edge.get(axis)
            if not need(isinstance(claim, Mapping), f'dependency {index}: {axis} mapping required'):
                continue
            status = claim.get('status')
            need(status in ('disjoint_established', 'jointly_modeled', 'not_applicable_established'),
                 f'dependency {index}: {axis} unresolved')
            need(pinned(claim.get('evidence')), f'dependency {index}: {axis} evidence pin required')
            if status == 'jointly_modeled':
                need(pinned(claim.get('adapter')) and text(claim.get('conditional_law_id')),
                     f'dependency {index}: {axis} conditional adapter required')
    for a, b in combinations(ids, 2):
        need(frozenset((a,b)) in indexed, f'{a}/{b}: missing dependence assessment')
    need(ledger.get('complete_vector_status') == 'bound' and pinned(ledger.get('vector_manifest')),
         'complete poll vector unbound')
    return result()
