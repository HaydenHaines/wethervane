"""Execute exact native observation method at invented fixed rates; never fit."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pymc as pm
import pytensor.tensor as pt

HERE = Path(__file__).resolve().parent
OWNER = Path('/home/hayden/projects/wethervane-redesign-20261006/integration')
SOURCE = OWNER / 'research/senate-stack-integration-20261020/modular-count-graph-astra-v1/components.py'
PROTOTYPE = HERE.parent / 'm01-count-measurement-prototype-sol-v1/prototype.py'

def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Measurement')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'observations')
    namespace = {'np': np}
    # Compile the original AST, with its original line positions. No rewritten law.
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    spec = importlib.util.spec_from_file_location('m01_supplied_state_prototype', PROTOTYPE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    covariance = np.array([[.12,.025,.01],[.025,.09,.018],[.01,.018,.14]])
    observed = np.array([.18,-.11,.42])
    # Different contests, candidate orders, and one reversed pair stress the real port.
    contests = {'a': {'candidates': [{'id':'d'}, {'id':'r'}]},
                'b': {'candidates': [{'id':'r'}, {'id':'d'}]}}
    rows = [dict(contest_id='a', first_candidate_id='d', second_candidate_id='r', observation_id='p1', logit_value=observed[0]),
            dict(contest_id='b', first_candidate_id='d', second_candidate_id='r', observation_id='p2', logit_value=observed[1]),
            dict(contest_id='a', first_candidate_id='r', second_candidate_id='d', observation_id='p3', logit_value=observed[2])]
    context = SimpleNamespace(source_names=[], contest_map=contests,
        data={'participation_reports': [], 'polls':[dict(id='synthetic', rows=rows, covariance=covariance.tolist())]})
    checks=[]
    for rates_a, rates_b in [([60.,40.],[45.,55.]), ([10.,90.],[80.,20.]), ([600.,400.],[450.,550.])]:
        claims=[]
        scope=SimpleNamespace(inputs={'category_rates': {'a':pt.as_tensor_variable(np.array(rates_a)), 'b':pt.as_tensor_variable(np.array(rates_b))}},
                              factor=lambda name,tokens: claims.append(dict(name=name,tokens=tokens)))
        with pm.Model() as model:
            namespace['observations'](None, context, scope)
        native=float(model.compile_logp()({}))
        mean=np.array([np.log(rates_a[0]/rates_a[1]),np.log(rates_b[1]/rates_b[0]),np.log(rates_a[1]/rates_a[0])])
        mixture=module.SharedVectorMixture(covariance[None,:,:],[1.])
        prototype=float(mixture.evaluate(observed,mean)[0])
        assert abs(native-prototype)<1e-10, (native,prototype)
        assert claims == [dict(name='poll_likelihood_synthetic',tokens=['poll:p1','poll:p2','poll:p3'])]
        checks.append(dict(ratesA=rates_a,ratesB=rates_b,mean=mean.tolist(),nativeLogp=native,prototypeLogp=prototype,absoluteError=abs(native-prototype),claims=claims))
    assert abs(checks[0]['nativeLogp']-checks[2]['nativeLogp'])<1e-10
    result=dict(schema='M01ExactNativeMethodSuppliedStateParityV1', status='passed',
        scope='Invented fixed-state arithmetic/interface check; no complete native graph, fit, sampling, calibration bank or historical replay.',
        pins=[pin(SOURCE),pin(PROTOTYPE),pin(Path(__file__)),pin(OWNER/'research/coordination-frontend-20261026/m01-count-binding-followup-v2.md')],
        nativeMethodLines=[method.lineno,method.end_lineno],checks=checks,
        limitations=['Scope token receipt checked locally; complete compiler evidence inventory not executed.',
            'Survey and participation-report branches deliberately empty; no claim about their laws.',
            'One synthetic poll block only; actual complete-vector assembly and cross-block dependence remain unvalidated.',
            'No gradients of the complete native model, protected-parent normalizers, actual posterior diagnostics or predictive skill established.',
            'Historical 2022/2018 A and 2024/2020 B replays require earlier-only admitted calibration banks, matched joint refits and owner runtime coordination.'])
    (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],checks=len(checks),maximumAbsoluteError=max(c['absoluteError'] for c in checks))))

if __name__ == '__main__':
    main()
