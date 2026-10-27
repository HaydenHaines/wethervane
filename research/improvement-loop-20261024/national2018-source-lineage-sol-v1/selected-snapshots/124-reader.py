"""Authenticated preparation reader; not execution or posterior admission."""
import json
from pathlib import Path
try:
 from . import assemble as a
except ImportError:
 import importlib.util,sys
 name='national_assembly_reader_helpers';spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name('assemble.py'));a=importlib.util.module_from_spec(spec);sys.modules[name]=a;spec.loader.exec_module(a)

def checked(record):
 if set(record)!={'path','sha256'}:raise ValueError('Exact file pin required')
 path=a.ROOT/record['path']
 if a.pin(path)!=record:raise ValueError('Changed prepared artifact: '+record['path'])
 return path

def load_prepared(index_pin,*,allow_incomplete_research=False):
 a.verify_pins();index=json.loads(checked(index_pin).read_bytes())
 if index['schema']!='NationalCountProblemPreparationV1' or index['fit_admitted'] or index['posterior_issued']:raise ValueError('Explicit preparation artifact required')
 expected={str(p.relative_to(a.ROOT)):a.pin(p)['sha256'] for p in a.H.glob('*.py') if p.name not in ['check.py','test_boundaries.py','finalize.py','test_reader.py']}
 expected[str((a.H/'source-pins.json').relative_to(a.ROOT))]=a.pin(a.H/'source-pins.json')['sha256']
 if {p['path']:p['sha256'] for p in index['implementation']}!=expected:raise ValueError('Wrong implementation binding')
 for pin in index['implementation']:checked(pin)
 if set(index['artifacts'])!={'problem','extension','parents','manifest','population_binding','construction_checks'}:raise ValueError('Complete preparation records required')
 data={k:json.loads(checked(v).read_bytes()) for k,v in index['artifacts'].items()}
 p=a.package().validate_problem(data['problem']);a.package().validate_mixed_extension(p,data['extension']);m=data['manifest'];binding=data['population_binding'];rows=data['parents']
 if p.identity!=index['problem_identity'] or len(m['settings'])!=35 or len({x['setting_id'] for x in m['settings']})!=35:raise ValueError('Exact full setting universe required')
 if m['national_complete'] is not True and allow_incomplete_research is not True:raise ValueError('Incomplete national research assembly; explicit partial preparation opt-in required')
 receipt=json.loads(checked(binding['receipt']).read_bytes())
 if binding['parent_ids']!=receipt['parent_ids'] or binding['masses']!=receipt['masses'] or binding['original_bank_identity']!=receipt['annual_identity']:raise ValueError('Changed original upstream mixture')
 if len(rows)!=8 or [x['parent_id'] for x in rows]!=binding['parent_ids'] or [x['mass'] for x in rows]!=binding['masses'] or [x['row_id'] for x in rows]!=list(range(8)):raise ValueError('All original whole parents required')
 for row in rows:
  if row['upstream_identity']!=binding['original_bank_identity'] or row['problem_identity']!=p.identity:raise ValueError('Changed parent target or source bank')
 # Each saved parent was validated by the source-bound producer; authenticated
 # bytes retain its exact exposure. No new population or parameter draws here.
 return dict(problem=p,parents=rows,extension=data['extension'],manifest=m,scope='incomplete conditional research preparation; no posterior/execution admission')
