"""Invented V2 completion fixture; tests schema acceptance vs numerical tolerance enforcement."""
import json,pathlib,subprocess,sys,tempfile,hashlib
PK=pathlib.Path(sys.argv[1]).resolve();sys.path.insert(0,str(PK))
import common as c,selection as s
phase='dynamic-witness';ordinal=2
with tempfile.TemporaryDirectory() as tmp:
 parent=pathlib.Path(tmp)/'parent-2';out=parent/phase
 p=subprocess.run([sys.executable,'-B',str(PK/'protocol_fixture.py'),str(out),phase],capture_output=True,text=True,timeout=10)
 assert p.returncode==0,p.stderr
 comp_path=parent/'dynamic-witness-comparison.json'
 comp=json.loads(comp_path.read_text())
 # Exact schema/status and finite, nonnegative values, but far above the design's comparison tolerance.
 comp['errors'][0]['logp']=1e100
 c.atomic_json(comp_path,comp)
 marker=out/'controller-success.json';receipt=json.loads(marker.read_text())
 receipt['comparison_artifacts']['../dynamic-witness-comparison.json']=c.pin(comp_path)
 c.atomic_json(marker,receipt)
 response=s.controller_response(out,ordinal,phase)
 retained=parent/'caller-return.json'
 c.atomic_json(retained,{'schema':'RetainedSelectedStageToolReturnV1','write_stdin_return':{'exit_code':0,'output':json.dumps(response)}})
 accepted=s.completed(out,retained,c.sha(retained),ordinal,phase)
 print(json.dumps({'status':'accepted_by_completed','phase':phase,'error_value':1e100,'schema_and_pins_coherently_rebuilt':True,'fixture_numeric_contents':'invented','same_user_coherent_rewrite':'yes'},sort_keys=True))
