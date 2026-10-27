"""Serial bounded isolated original/prototype measurements; never adopt stores."""
from pathlib import Path
import json,shutil,subprocess,tempfile
HERE=Path(__file__).resolve().parent;ORIGINAL=HERE.parent/'queue-finite-ancestor-sol-v1'
outputs={}
for label,source in [('original',ORIGINAL),('prototype',HERE)]:
 with tempfile.TemporaryDirectory(prefix='decode-measure-'+label+'-')as name:
  for p in source.glob('*.py'):shutil.copy2(p,Path(name)/p.name)
  shutil.copy2(HERE/'measure_child.py',Path(name)/'measure_child.py')
  result=subprocess.run(['nice','-n','15','python3','-B','measure_child.py'],cwd=name,capture_output=True,text=True,timeout=15)
  if result.returncode:raise RuntimeError('measurement failed: '+result.stderr)
  outputs[label]=json.loads(result.stdout)
print(json.dumps(outputs,indent=2))
