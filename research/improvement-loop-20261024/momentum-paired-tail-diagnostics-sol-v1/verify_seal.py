"""Read-only byte verification, no arithmetic execution or data loading."""
import hashlib
import json
from pathlib import Path
here=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seal=json.loads((here/'seal.json').read_text())
assert sha(here/'artifact-manifest.json')==seal['manifest_sha256']
manifest=json.loads((here/'artifact-manifest.json').read_text())
for entry in manifest['files']:assert sha(here/entry['path'])==entry['sha256'],entry['path']
for entry in json.loads((here/'SOURCES.json').read_text())['bindings']:assert sha(Path(entry['path']))==entry['sha256'],entry['path']
print(json.dumps(dict(status='passed',manifest_sha256=seal['manifest_sha256'],files=len(manifest['files']))))
