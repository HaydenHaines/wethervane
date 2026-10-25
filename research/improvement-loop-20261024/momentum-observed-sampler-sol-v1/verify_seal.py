"""Read-only byte verification; no sampler, tests, labels, or RNG calls."""
from pathlib import Path
import hashlib
import json

here=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seal=json.loads((here/'seal.json').read_text())
assert sha(here/'MANIFEST.json')==seal['manifest_sha256']
manifest=json.loads((here/'MANIFEST.json').read_text())
for entry in manifest['files']:
    assert sha(here/entry['path'])==entry['sha256'],entry['path']
for entry in json.loads((here/'SOURCES.json').read_text())['local_import_sources']:
    assert sha(Path(entry['owner_path']))==entry['sha256'],entry['owner_path']
print(json.dumps(dict(status='passed',files=len(manifest['files']),manifest_sha256=seal['manifest_sha256'])))
