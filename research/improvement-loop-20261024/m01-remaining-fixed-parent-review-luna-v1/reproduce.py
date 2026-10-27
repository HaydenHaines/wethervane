"""Small read-only challenge fixtures for frozen remaining-fixed-parent packet."""
import json, pathlib, tempfile
import common, selection, compare
c=common.read_contract()
# Ensure complete selected-parent identity includes exposure bytes: a changed exposure changes record digest.
record=common.binding if False else None
b=compare.base.binding(c)
full=b.load()['parents'][2]['record']
contract={
 'composition_identity':c['reference_composition_identity'],
 'problem_identity':c['problem_identity'],
 'parent_identity':common.digest(full),
 'parent_id':full['parent_id'],
 'upstream_identity':full['upstream_identity'],
 'parent_mass':.125,
 'parent_mass_used_in_logp':False,
 'inference_target':'population_cut_v1',
 'scientific_target_id':compare.target_identity(full,c),
}
assert compare.validate_fixed_target(contract,full,c)
changed=json.loads(json.dumps(full)); changed['exposure'][0][0]+=1e-8
rejected=False
try:
 bad=dict(contract,parent_identity=common.digest(changed),scientific_target_id=compare.target_identity(changed,c))
 compare.validate_fixed_target(bad,full,c)
except ValueError:
 rejected=True
assert rejected
# Challenge the new-stage acknowledgement helper: a caller-supplied hash over a self-authored
# success-shaped receipt is accepted without the retained write_stdin_return JSON chain.
with tempfile.TemporaryDirectory() as tmp:
 out=pathlib.Path(tmp)/'dynamic-witness';out.mkdir()
 result={'status':'completed_calls_not_normalized_conditionals','selected_parent_ordinal':2,'phase':'dynamic-witness','contract_sha256':common.EXPECTED_CONTRACT}
 terminal={'arm':'dynamic-witness','returncode':0,'reason':None}
 common.atomic_json(out/'result.json',result);common.atomic_json(out/'terminal.json',terminal)
 receipt={'status':'successfully_reaped_selected_phase_and_passed_comparison','contract_sha256':common.EXPECTED_CONTRACT,'selected_parent_ordinal':2,'phase':'dynamic-witness','result':common.pin(out/'result.json'),'terminal':common.pin(out/'terminal.json'),'comparison_artifacts':{}}
 common.atomic_json(out/'controller-success.json',receipt)
 supplied_digest=common.sha(out/'controller-success.json')
 accepted=selection.completed(out,supplied_digest,2,'dynamic-witness')
 assert accepted['status']=='successfully_reaped_selected_phase_and_passed_comparison'
print(json.dumps({'status':'passed','changed_exposure_identity_rejected':rejected,'self_authored_success_receipt_accepted_by_completed_helper':True,'no_graph_or_model_imported':True},sort_keys=True))
