"""Authenticate frozen emission and finite ranked implementations before reuse."""
import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PINS={'survey-answer-emission-astra-v1':'7e34a77875f432d4b3150bdc7554f2dfa948db37454dc5517a12ddaf99bda872',
      'ranked-runoff-graph-astra-v1':'0c5934afb52387001ffbac5eda31fc3a7265a8a8b4fce596ce5ccd6bbef04b27'}
def package(folder):
    path=HERE.parent/folder;receipt=path/'receipt-v1.json'
    if sha256(receipt.read_bytes()).hexdigest()!=PINS[folder]:raise ValueError('Frozen receipt changed')
    for r in json.loads(receipt.read_bytes())['files']:
        if sha256((ROOT/r['path']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Frozen evidence/source changed: '+r['path'])
    name='rank_report_frozen_'+folder.replace('-','_')
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path/'package.py',submodule_search_locations=[str(path)])
        m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name]
emission_package=package('survey-answer-emission-astra-v1')
legacy_inputs=sys.modules[emission_package.__name__+'.inputs_v2']
rank_package=package('ranked-runoff-graph-astra-v1')
ranking=sys.modules[rank_package.__name__+'.ranking']
rank_joint=sys.modules[rank_package.__name__+'.joint']
