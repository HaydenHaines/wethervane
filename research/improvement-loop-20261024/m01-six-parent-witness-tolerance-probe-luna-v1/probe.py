"""Invented batch protocol only: challenge tolerance checks in completion.validate."""
import json,pathlib,subprocess,sys,tempfile
PK=pathlib.Path(sys.argv[1]).resolve();sys.path.insert(0,str(PK))
import common as c,completion as done
with tempfile.TemporaryDirectory() as tmp:
 root=pathlib.Path(tmp);out=root/'dynamic-six'
 proc=subprocess.run([sys.executable,'-B',str(PK/'protocol_fixture.py'),str(out)],capture_output=True,text=True,timeout=10)
 assert proc.returncode==0,proc.stderr
 comparison_path=root/'all-six-comparison.json';comparison=json.loads(comparison_path.read_text());comparison['parents'][0]['errors'][0]['logp']=1e100;c.atomic_json(comparison_path,comparison)
 marker=out/'controller-success.json';receipt=json.loads(marker.read_text());receipt['comparison_artifacts']['../all-six-comparison.json']=c.pin(comparison_path);c.atomic_json(marker,receipt)
 tool=root/'caller.json';c.atomic_json(tool,{'schema':'RetainedSelectedStageToolReturnV1','write_stdin_return':{'exit_code':0,'output':json.dumps(done.response(out))}})
 done.completed(out,tool,c.sha(tool))
 print(json.dumps({'status':'accepted_by_completed','error_value':1e100,'fixture':'coherently self-authored invented protocol output','actual_graph':False},sort_keys=True))
