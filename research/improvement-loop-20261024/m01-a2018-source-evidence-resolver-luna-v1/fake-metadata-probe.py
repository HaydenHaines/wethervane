"""Scratch-only false-metadata boundary probe. Never prints row values or identifiers."""
import copy, hashlib, json
from pathlib import Path
import resolver as r

bindings = r.load_bindings()
problem = json.loads(r.read_pinned(bindings["problem"]))
mutated = copy.deepcopy(problem)
mutated["polls"][0]["rows"][0]["logit_value"] += 0.001
inspected = r.inspect_prepared(mutated)
raw_mutated = json.dumps(mutated, separators=(",", ":"), ensure_ascii=False).encode()
def reader(path):
    original = Path(path).read_bytes()
    return raw_mutated if path == bindings["problem"]["path"] else original
try:
    r.resolve(reader)
    pinned_resolution = "unexpected_accept"
except r.Rejected as exc:
    pinned_resolution = str(exc)
# Invalid candidate targets are rejected before any output; do not print the target.
bad = copy.deepcopy(problem)
bad["polls"][0]["rows"][0]["first_candidate_id"] = "invented-outside-target"
try:
    r.inspect_prepared(bad)
    bad_target = "unexpected_accept"
except r.Rejected as exc:
    bad_target = str(exc)
print(json.dumps({
    "untrusted_inspector_accepts_well_formed_changed_value": bool(inspected["poll_ordered_vector_covariance_sha256"]),
    "pinned_resolver_rejects_changed_source_bytes": pinned_resolution,
    "inspector_rejects_out_of_contest_candidate": bad_target,
    "row_ids_or_values_emitted": False
}, sort_keys=True))

