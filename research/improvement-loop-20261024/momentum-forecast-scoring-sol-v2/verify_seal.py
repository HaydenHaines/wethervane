"""Read-only source/evidence byte verification; no prediction or label loading."""
import hashlib
import json
from pathlib import Path

here=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seal=json.loads((here/'seal.json').read_text())
assert sha(here/'artifact-manifest.json')==seal['manifest_sha256']
manifest=json.loads((here/'artifact-manifest.json').read_text())
for item in manifest['files']:assert sha(here/item['path'])==item['sha256'],item['path']
for item in json.loads((here/'SOURCES.json').read_text())['bindings']:assert sha(Path(item['path']))==item['sha256'],item['path']
print(json.dumps(dict(status='passed',manifest_sha256=seal['manifest_sha256'],files=len(manifest['files']))))
