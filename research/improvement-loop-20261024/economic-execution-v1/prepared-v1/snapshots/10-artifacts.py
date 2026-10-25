"""Content-bound local artifacts using the existing exclusive-write convention.

A hash-bound reference detects later mutation. It does not authenticate the
scientific evidence or supply an independently witnessed publication timestamp.
"""

import hashlib
import json
import re
from pathlib import Path

from .contracts import Digest, Name, Record


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def byte_digest(value):
    return hashlib.sha256(value).hexdigest()


class ArtifactRef(Record):
    artifact_id: Digest
    kind: Name


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root)

    @staticmethod
    def _name(name):
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
            raise ValueError("Flat artifact filename required")
        if name == "manifest.json":
            raise ValueError("Reserved manifest name")

    def put(self, kind, metadata, files):
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("Artifact kind required")
        for name, data in files.items():
            self._name(name)
            if type(data) is not bytes:
                raise ValueError("Exact bytes required")
        manifest = {
            "schema": "model-stack-artifact-v1",
            "kind": kind,
            "metadata": metadata,
            "files": {n: {"sha256": byte_digest(b), "bytes": len(b)} for n, b in sorted(files.items())},
        }
        payload = canonical(manifest)
        ref = ArtifactRef(artifact_id=byte_digest(payload), kind=kind)
        directory = self.root / ref.artifact_id
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir(exist_ok=False)
        except FileExistsError:
            # Never silently repair a partial attempt or overwrite different bytes.
            self.read(ref)
            return ref
        for name, data in sorted(files.items()):
            with (directory / name).open("xb") as stream:
                stream.write(data)
        with (directory / "manifest.json").open("xb") as stream:
            stream.write(payload)
        return ref

    def read(self, ref):
        if not isinstance(ref, ArtifactRef):
            raise ValueError("Hash-bound artifact reference required")
        ref = ArtifactRef.model_validate(ref)
        directory = self.root / ref.artifact_id
        if directory.is_symlink():
            raise ValueError("Artifact directory cannot be a symlink")
        path = directory / "manifest.json"
        if path.is_symlink():
            raise ValueError("Manifest cannot be a symlink")
        payload = path.read_bytes()
        if byte_digest(payload) != ref.artifact_id:
            raise ValueError("Artifact manifest changed")
        manifest = json.loads(payload)
        if (
            set(manifest) != {"schema", "kind", "metadata", "files"}
            or manifest["schema"] != "model-stack-artifact-v1"
            or manifest["kind"] != ref.kind
        ):
            raise ValueError("Artifact type/schema mismatch")
        if set(p.name for p in directory.iterdir()) != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Unregistered artifact files")
        files = {}
        for name, record in manifest["files"].items():
            self._name(name)
            path = directory / name
            if path.is_symlink():
                raise ValueError("Artifact files cannot be symlinks")
            value = path.read_bytes()
            if record != {"sha256": byte_digest(value), "bytes": len(value)}:
                raise ValueError("Artifact payload changed")
            files[name] = value
        return manifest["metadata"], files
