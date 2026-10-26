"""Validate supplied ownership claims, never authenticate them or run inference.

All relevant respondent/dependence edges must be explicitly inventoried by the
audit author. Passing this schema is necessary, not evidence of completeness.
"""
from datetime import datetime
from itertools import combinations


def validate(ledger):
    errors = []
    def need(ok, message):
        if not ok:
            errors.append(message)
    def pinned(value):
        return (isinstance(value, dict) and bool(value.get('path'))
                and isinstance(value.get('sha256'), str) and len(value['sha256']) == 64
                and all(c in '0123456789abcdef' for c in value['sha256']))
    def instant(value):
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt if dt.tzinfo else None
        except (ValueError, TypeError, AttributeError):
            return None
    need(ledger.get('mode') in ('historical_replay', 'retrospective_joint'), 'unsupported mode')
    cutoff = instant(ledger.get('cutoff'))
    need(cutoff is not None, 'timezone-aware cutoff required')
    records = ledger.get('records', [])
    ids = [r.get('id') for r in records]
    need(bool(ids) and all(isinstance(x, str) and x for x in ids) and len(set(ids)) == len(ids), 'unique record IDs required')
    atoms = {}
    training_labels = set()
    label_owners = {}
    target_labels = set(ledger.get('evaluation_event_keys', []))
    for r in records:
        rid = r.get('id')
        need(pinned(r.get('source')), f'{rid}: pinned source required')
        need(bool(r.get('factor_owner')), f'{rid}: numerical factor owner required')
        need(r.get('canonical_atoms_status') == 'established' and bool(r.get('canonical_atoms')), f'{rid}: canonical atoms unresolved')
        for atom in r.get('canonical_atoms', []):
            need(atom not in atoms, f'duplicate evidence atom {atom}')
            atoms[atom] = rid
        labels = r.get('outcome_label_event_keys')
        need(isinstance(labels, list), f'{rid}: explicit outcome label inventory required')
        for label in set(labels or []):
            need(label not in label_owners, f'outcome label reused across factors {label}')
            label_owners[label] = rid
        training_labels.update(labels or [])
        if ledger.get('mode') == 'historical_replay':
            available = instant(r.get('verified_available_at'))
            need(r.get('availability_status') == 'verified' and available is not None,
                 f'{rid}: historical availability unresolved; field end is not publication')
            if available is not None and cutoff is not None:
                need(available <= cutoff, f'{rid}: available after cutoff')
            need(r.get('historical_version_status') == 'verified', f'{rid}: historical version unresolved')
    need(not (training_labels & target_labels), 'evaluation outcome used as training input')
    # The author must assess all record pairs for respondent reuse and shared
    # source/error dependence; disjoint atom IDs alone do not establish either.
    edges = ledger.get('dependencies', [])
    indexed = {}
    for edge in edges:
        pair = frozenset(edge.get('records', []))
        need(len(pair) == 2 and pair <= set(ids), 'unknown dependency record')
        need(pair not in indexed, 'duplicate dependency assessment')
        indexed[pair] = edge
    for a,b in combinations(ids, 2):
        edge = indexed.get(frozenset((a,b)))
        if edge is None:
            errors.append(f'{a}/{b}: missing dependence assessment')
            continue
        for axis in ('respondents', 'source_error'):
            claim = edge.get(axis, {})
            status = claim.get('status')
            need(status in ('disjoint_established', 'jointly_modeled', 'not_applicable_established'), f'{a}/{b}: {axis} unresolved')
            need(pinned(claim.get('evidence')), f'{a}/{b}: {axis} evidence pin required')
            if status == 'jointly_modeled':
                need(pinned(claim.get('adapter')) and bool(claim.get('conditional_law_id')),
                     f'{a}/{b}: {axis} conditional adapter required')
    need(ledger.get('complete_vector_status') == 'bound' and pinned(ledger.get('vector_manifest')), 'complete poll vector unbound')
    return dict(status='contract_checks_passed' if not errors else 'unresolved_or_rejected',
                errors=errors, authenticated=False, fit_authorized=False,
                qualification='Checks supplied claims only; source authentication, completeness and scientific review remain separate.')
