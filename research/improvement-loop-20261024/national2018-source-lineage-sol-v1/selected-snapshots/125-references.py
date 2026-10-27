"""Read-only pinned mathematical helpers and equivalence references."""

import hashlib
import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def load(name, path, sha):
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError("Frozen reference dependency changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cycle = load(
    "modular_count_cycle_helper_v1",
    BASE / "cycle-separation-astra-v1/cycle_coordinates.py",
    "ba6332df1774d5e8fc260b8dc0d0d6aac04affdf699de11ee2177065c92c1f7f",
)
roles = load(
    "modular_count_role_binding_v1",
    BASE / "candidate-role-hierarchy-astra-v1/roles.py",
    "946f98108658081792dcefe957fe1dc0ca5dc929224a84e55f7e59f0559b7721",
)


def baseline_module(role_hierarchy=False):
    if role_hierarchy:
        return load(
            "modular_count_role_equivalence_reference_v1",
            BASE / "cycle-separation-astra-v1/joint_role_cycle_separated.py",
            "06a3351cf7bf05ef15599af4c5224ba1fd5a427b2f3df21814d7a3bfdf439bfa",
        )
    return load(
        "modular_count_equivalence_reference_v1",
        BASE / "cycle-separation-astra-v1/joint_cycle_separated.py",
        "76fed40cd015d08511c230c1fe567b5ce886b803547fcfe3b96ebf74e93a9109",
    )
