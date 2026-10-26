"""Source-derived actual symbolic pipeline. Imported only after technical/resource gates.
No actual graph is executed by tests; exact scientific guards remain source-pinned.
"""
import hashlib,json,sys,time,traceback
from pathlib import Path
from common import checked,need
from count_core import count_bundle

def digest(raw):return hashlib.sha256(raw).hexdigest()
def read_pin(item):return checked(item)

def symbolic(plan, receipt, output, writer):
    import numpy as np
    import pytensor
    import pytensor.tensor as pt
    from pytensor.graph.replace import graph_replace
    from pytensor.graph.traversal import ancestors, graph_inputs
    from pytensor.raise_op import CheckAndRaise
    from pymc.pytensorf import rewrite_pregrad, gradient
    out = Path(output)
    records = []

    def persist(name, detail):
        records.append(dict(stage=name, detail=detail, unix=time.time()))
        writer.json('stages.json', dict(schema='SymbolicCountStagesV1', records=records))
        print(name, flush=True)
    try:
        persist('execution_gate_passed', dict(plan_scope=plan['scope'], numerical_compile=False))
        old = Path(plan['frozen_provider_directory'])
        sys.path.append(str(old / 'schema-overlay'))
        sys.path.insert(0, str(old))
        import provider as p
        s = p.selection()
        problem = json.loads(Path(s['problem_path']).read_text())
        parent = s['parents'][0]
        p.validate(problem, parent, s)
        roles_path = p.OWNER / 'research/senate-stack-integration-20261020/candidate-role-source-sol-v1/prepared-v2/candidate-role-manifest.json'
        rm = json.loads(roles_path.read_text())
        original = json.loads((p.OWNER / rm['bound_prepared_problem']['path']).read_text())
        refs = sys.modules['m01_successor_native.references']
        roles = refs.roles.load_role_binding(original, roles_path, 'a1d73f7dece21a4cb5b104c69caec5157e4dd106b411ce17833f7f5a41bbeb3d', admission_cutoff_utc=problem['cutoff_utc'], root=p.OWNER)
        roles = refs.roles.rebind_role_binding(original, problem, roles)
        registry = p.components.standard_registry()
        successor = p.M01Measurement('actual_calibration_bank')
        registry.register(successor)
        composition = p.components.baseline_composition(role_hierarchy=True)
        composition = p.graph.Composition(tuple(((r, successor.implementation_id if r == 'measurement' else n) for r, n in composition.selections)), composition.standardized, composition.person_coordinates)
        model = p.graph.compile_model(registry, composition, problem, parent, roles=roles)
        contract = p.dict_plain(model.joint_contract)
        inventory = p.dict_plain(model.component_inventory)
        frozen_inventory = json.loads(read_pin(plan['frozen_factor_inventory']))
        need(inventory == frozen_inventory, 'factor/parameter/token ownership changed')
        need(contract['parent_identity'] == p.graph.digest(parent) and contract['parent_mass'] == 0.125 and (contract['parent_mass_used_in_logp'] is False) and (contract['inference_target'] == 'population_cut_v1'), 'parent/cut contract changed')
        need(contract['scientific_target_id'] == p.graph.digest(dict(composition_id=contract['composition_identity'], problem=contract['problem_identity'], parent=p.graph.digest(parent), target='population_cut_v1')), 'scientific target mismatch')
        writer.text('first-parent-contract.json', json.dumps(contract, indent=2) + '\n')
        writer.text('factor-inventory.json', json.dumps(inventory, indent=2) + '\n')
        persist('original_graph_and_first_parent_authenticated', dict(contract=contract, inventory_sha256=digest(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode())))
        ordered = list(model.continuous_value_vars)
        ordered_set = set(ordered)
        need(len(ordered) == len(model.value_vars), 'discrete or omitted coordinates unsupported')
        need([rv.name for rv in model.free_RVs] == contract['required_free_rvs'], 'original free-RV order changed')
        witness = json.loads(read_pin(plan['frozen_shape_witness']))
        shapes = next((c for c in witness['cases'] if c['case'] == 'actual_TX_role_graph_only'))['modular']['parameter_shapes']
        need({v.name for v in ordered} == set(shapes), 'full transformed variable names changed')
        segments = []
        offset = 0
        for v in ordered:
            shape = shapes[v.name]
            need(v.dtype == 'float64' and v.ndim == len(shape), 'coordinate type/rank mismatch')
            need(all((n is None or n == m for n, m in zip(v.type.shape, shape))), 'static coordinate shape mismatch')
            size = int(np.prod(shape, dtype=int)) if shape else 1
            segments.append(dict(name=v.name, shape=shape, dtype=v.dtype, start=offset, stop=offset + size))
            offset += size
        need(offset == 662, 'full coordinate dimension changed')
        writer.text('coordinate-order.json', json.dumps(segments, indent=2) + '\n')
        persist('exact_native_continuous_value_order_saved', dict(dimensions=offset, variables=len(ordered), order='model.continuous_value_vars; native gradient helper flatten/concat'))
        old_port = model.typed_ports['exposure']
        need(hasattr(old_port, 'data') and np.array_equal(old_port.data, np.asarray(parent['exposure'])), 'exact original exposure constant required')
        cost = model.logp(jacobian=True)
        before = set(ancestors([cost]))
        need(old_port in before, 'exact exposure identity absent from original complete logp')
        E = pt.dmatrix('authenticated_whole_parent_exposure')
        replacement = pt.specify_shape(E, (3, 24))
        replacement = CheckAndRaise(ValueError, 'positive finite whole-parent exposure required')(replacement, pt.all((replacement > 0) & ~pt.isnan(replacement) & ~pt.isinf(replacement)))
        need(E not in ordered_set, 'conditioning input cannot become a sampled coordinate')
        replaced = graph_replace(cost, {old_port: replacement}, strict=True)
        original_inputs = {v for v in graph_inputs([cost]) if v in ordered_set}

        def structure(label, expression):
            nodes = set(ancestors([expression]))
            inputs = set(graph_inputs([expression]))
            need(E in inputs, label + ': explicit E identity/dependence lost')
            need(old_port not in nodes, label + ': original exposure node retained')
            stale = []
            for node in nodes:
                if hasattr(node, 'data'):
                    data = np.asarray(node.data)
                    if data.shape == (3, 24) and any((np.array_equal(data, np.asarray(row['exposure'])) for row in s['parents'])):
                        stale.append(str(node))
            need(not stale, label + ': stale whole-parent constant detected')
            return dict(expression=label, ancestor_variables=len(nodes), explicit_E_input=True, old_port_absent=True, stale_whole_parent_arrays=0, value_input_names=sorted((v.name for v in inputs if v in ordered_set)))
        pre = structure('complete_replaced_logp', replaced)
        need({v for v in graph_inputs([replaced]) if v in ordered_set} == original_inputs, 'original value inputs changed during replacement')
        persist('strict_logp_replacement_passed', pre)
        rewritten = rewrite_pregrad(replaced)
        post = structure('rewrite_pregrad_replaced_logp', rewritten)
        need({v for v in graph_inputs([rewritten]) if v in ordered_set} == original_inputs, 'value input set changed under pregradient rewrite')
        persist('native_pregradient_rewrite_checked', post)
        derived = pt.specify_shape(gradient(rewritten, ordered), (662,))
        final = structure('native_order_derived_gradient', derived)
        need(derived.ndim == 1, 'gradient must be one vector')
        persist('gradient_after_replacement_checked', final)
        need(p.dict_plain(model.component_inventory) == inventory, 'symbolic work mutated factors or ownership')
        import pickle
        counted = count_bundle(dict(original_cost=cost, replaced_cost=replaced, rewritten_cost=rewritten, gradient=derived, ordered_value_vars=ordered, E=E, segments=segments), writer, plan['count_budget_bytes'])
        result = dict(schema='ActualTXSymbolicStreamCountV1', status='symbolic_stream_counted_not_numeric_or_restored', first_parent_contract=contract, checks=[pre, post, final], coordinate_segments=segments, dimensions=662, gradient_path='model.logp(jacobian=True) -> strict identity exposure replace -> pinned pymc.rewrite_pregrad -> pinned pymc.gradient(cost,model.continuous_value_vars)', mass_argument=False, numerical_compilation=False, numerical_point_saved=False, numerical_gradient_saved=False, scientific_qualification=False, notes=['Exposure remains conditioning input only.', 'No second-parent target evaluated.', 'Absence of equal whole exposure constants is a diagnostic, not a full algebraic independence proof.', 'Complete future numeric persistence contract is in plan.json.'], stream_count=counted, pickle_payload_persisted=False, pickle_loaded=False)
        writer.text('result.json', json.dumps(result, indent=2) + '\n')
        persist('completed_symbolic_only', dict(status=result['status']))
    except BaseException:
        writer.text('failure.log', traceback.format_exc())
        persist('failed_closed', dict(error=traceback.format_exc()))
        raise
