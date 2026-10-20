#!/usr/bin/env python3
"""Export reviewed provider metadata, without activating results or an executor.

Only the pinned Senate v2 handoff is admitted. Reads JSON and checks existing
file bytes; never imports scientific drivers or executes model calculations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

PACKET = Path("research/fundamentals-iteration-20261020/frontend-contract-root-v2")
INTERFACE_SHA256 = "85f38310f650fc7eb0b18739f36e36a3a7457602cb5e42f328a89f1b58bb1ece"
READY_SHA256 = "367dae36f2f6f79c089a96524f369867971f1cec21537e7b542f59a70ed2f803"
TARGET = "senate_aggregate_logit_empirical_mixture_v1"
CONTRACT = "empirical_linear_fundamentals_joint_v1"
CUTOFF = "2018-10-23T00:00:00Z"
ROLE_COPY = {
    "population": ("Population and geography", "Who and which places the model describes."),
    "community": ("Community", "How measured groups and their composition are represented."),
    "fundamentals": (
        "Electoral starting estimates",
        "Electoral history, candidate adjustments and race-specific variation.",
    ),
    "environment": (
        "Shared political environment",
        "National patterns and uncertainty that connect races. Multiple providers can contribute together.",
    ),
    "measurement": ("Measurements such as polls", "How observations inform the model, including their errors."),
    "participation": ("Participation", "Who participates in an election or survey."),
    "choice": ("Candidate choice", "The eligible choices and the conditions under which people choose."),
    "counts_rules": ("Vote counts and election rules", "How votes, winners and chamber outcomes are derived."),
}


class CatalogError(ValueError):
    """The source packet differs from the reviewed contract."""


def require(condition, message):
    if not condition:
        raise CatalogError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def pinned(root, reference):
    relative = Path(reference["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe source reference")
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), "Source reference leaves checkout")
    require(path.is_file() and sha(path) == reference["sha256"], f"Source fingerprint mismatch: {relative.name}")
    return path


def request_digest(request):
    canonical = copy.deepcopy(request)
    canonical["selections"].sort(key=lambda choice: choice["instance_id"])
    return digest(canonical)


def configuration_digest(configuration):
    canonical = copy.deepcopy(configuration)
    canonical["request"] = request_digest(canonical["request"])
    canonical["dependencies"].sort(key=lambda pin: pin["path"])
    canonical["outputs"].sort(key=lambda port: port["port"])
    return digest(canonical)


def validate_option_schema(schema):
    require(
        schema.get("type") == "object" and schema.get("additionalProperties") is False,
        "Provider options must be a closed declarative object",
    )
    require(isinstance(schema.get("properties"), dict), "Provider options have no properties")
    # This reviewed snapshot uses only string enums. Fail closed on new editor
    # kinds or references; a later source-bound adapter can admit them explicitly.
    require(
        set(schema) <= {"type", "description", "properties", "additionalProperties"}, "Unknown option-schema keyword"
    )
    for option in schema["properties"].values():
        require(set(option) <= {"type", "enum", "default", "description"}, "Unknown option keyword")
        values = option.get("enum")
        require(
            option.get("type") == "string"
            and isinstance(values, list)
            and values
            and all(isinstance(value, str) for value in values)
            and len(values) == len(set(values))
            and option.get("default") in values,
            "Invalid string-enum option/default",
        )


def validate_ready(ready, request, arm):
    require(ready["status"] == "ready", "Example result is not ready")
    configuration, record, summary = ready["configuration"], ready["record"], ready["summary"]
    require(configuration["request"] == request, "Ready request differs from pinned example")
    require(ready["request_sha256"] == request_digest(request), "Ready request identity mismatch")
    identity = configuration_digest(configuration)
    require(
        record["configuration_sha256"] == summary["configuration_sha256"] == arm["configuration_sha256"] == identity,
        "Ready configuration identity mismatch",
    )
    require(ready["artifact"] == arm["artifact"], "Ready artifact identity mismatch")
    require(
        record["numerical_status"] == "passed" and summary["numerical_passed"] is True,
        "Example numerical checks did not pass",
    )
    require(
        record["evidence_status"] == summary["evidence_status"] == arm["evidence_status"] == "unassessed",
        "Example evidence qualification changed",
    )
    ids = summary["race_ids"]
    require(
        len(ids) == len(set(ids)) == 30 and all(race.startswith("2018:") for race in ids),
        "Example support differs from ordinary 2018 subset",
    )
    require(summary["quantile_levels"] == [0.025, 0.1, 0.5, 0.9, 0.975], "Example interval levels changed")
    require(
        len(summary["share_quantiles"]) == 5 and all(len(row) == 30 for row in summary["share_quantiles"]),
        "Example share quantile shape changed",
    )
    for values in [summary["dem_win_probability"], summary["mean_dem_share"], *summary["share_quantiles"]]:
        require(
            len(values) == 30 and all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in values),
            "Example has invalid probabilities",
        )
    for i in range(30):
        require(
            all(summary["share_quantiles"][q][i] <= summary["share_quantiles"][q + 1][i] for q in range(4)),
            "Example quantiles are not ordered",
        )
    require(
        request["target_id"] == summary["target_id"] == TARGET
        and request["output_contract_id"] == CONTRACT
        and request["information_cutoff"] == CUTOFF,
        "Example target/contract/cutoff mismatch",
    )


def export_catalog(root):
    interface_path = root / PACKET / "interface-v1/receipt.json"
    ready_path = root / PACKET / "ready-v1/receipt.json"
    require(
        sha(interface_path) == INTERFACE_SHA256 and sha(ready_path) == READY_SHA256,
        "Receipt differs from reviewed Senate v2 handoff",
    )
    interface, receipt = read(interface_path), read(ready_path)
    require(
        interface["scheduler_available"] is False
        and receipt["status"] == "passed"
        and receipt["default_changed"] is False
        and receipt["target_scoring"] is False,
        "Receipt lifecycle or evidence boundaries changed",
    )
    output_paths = {Path(ref["path"]).name: pinned(root, ref) for ref in interface["outputs"]}
    for reference in interface["driver_sources"]:
        pinned(root, reference)
    ready_paths = {Path(ref["path"]).name: pinned(root, ref) for ref in receipt["outputs"]}
    index_path = pinned(root, receipt["result_index"])
    index = read(index_path)
    catalog = read(output_paths["catalog.json"])
    require(catalog["executable_targets"] == [TARGET], "Unexpected executable target")
    require([r["role"] for r in catalog["roles"]] == list(ROLE_COPY), "Expected all eight reviewed roles")
    roles, provider_ids, instances = [], set(), set()
    for role in catalog["roles"]:
        providers = []
        for offer in role["providers"]:
            require(offer["role"] == role["role"] and offer["target_ids"] == [TARGET], "Provider role/target mismatch")
            require(
                offer["provider_id"] not in provider_ids and offer["default_instance_id"] not in instances,
                "Duplicate provider or default instance",
            )
            provider_ids.add(offer["provider_id"])
            instances.add(offer["default_instance_id"])
            validate_option_schema(offer["options_schema"])
            for source in offer["sources"]:
                pinned(root, source)
            providers.append(
                {
                    key: copy.deepcopy(offer[key])
                    for key in (
                        "provider_id",
                        "version",
                        "default_instance_id",
                        "role",
                        "label",
                        "availability",
                        "options_schema",
                        "inputs",
                        "outputs",
                        "target_ids",
                        "limitations",
                    )
                }
            )
            providers[-1].update(
                siteAvailability="metadata_only", sourceFingerprints=[source["sha256"] for source in offer["sources"]]
            )
        label, description = ROLE_COPY[role["role"]]
        roles.append(
            dict(
                role=role["role"],
                label=label,
                description=description,
                availability="metadata_only" if providers else "unavailable",
                providers=providers,
            )
        )
    require(len(provider_ids) == 7, "Expected seven concurrent provider instances")
    examples, lifecycle, contexts = [], [], []
    for arm in index["arms"]:
        name = arm["arm"]
        require(name in {"F", "FP"}, "Unexpected saved example")
        request = read(output_paths[f"{name}-request.json"])
        ready_file = ready_paths[f"{name}-ready.json"]
        ready = read(ready_file)
        validate_ready(ready, request, arm)
        for dependency in ready["configuration"]["dependencies"]:
            pinned(root, dependency)
        manifest_path = pinned(root, arm["manifest"])
        manifest = read(manifest_path)
        require(sha(manifest_path) == ready["artifact"]["artifact_id"], "Artifact manifest identity mismatch")
        for filename, file in manifest["files"].items():
            require(Path(filename).name == filename, "Unsafe artifact filename")
            require(sha(manifest_path.parent / filename) == file["sha256"], "Immutable result file changed")
        require(read(pinned(root, arm["summary"])) == ready["summary"], "Ready summary differs from immutable result")
        boundary_path = pinned(root, arm["negative_boundaries"])
        boundary = read(boundary_path)
        pending = boundary["changed_candidate_resolution"]
        changed = boundary["changed_candidate_request"]
        choice = next(c for c in changed["selections"] if c["instance_id"] == "candidate")
        require(
            choice["options"] == {"candidate": "off"}
            and pending["status"] == "requires_computation"
            and pending.get("artifact") is None
            and pending.get("summary") is None,
            "Candidate-off lifecycle boundary changed",
        )
        unsupported = boundary["unavailable_target_resolution"]
        require(
            unsupported["status"] == "incompatible"
            and unsupported.get("artifact") is None
            and unsupported.get("summary") is None,
            "Unsupported target returned a result",
        )
        # Full executable source paths remain server-side. These templates are
        # explicit UI drafts and fingerprints, not directly executable requests.
        template = {
            key: copy.deepcopy(request[key])
            for key in ("target_id", "information_cutoff", "output_contract_id", "selections")
        }
        context_id = f"ordinary-senate-2018-{name.lower()}"
        contexts.append(
            dict(
                id=context_id,
                year=2018,
                supportId=ready["configuration"]["support_id"],
                raceIds=ready["summary"]["race_ids"],
                raceCount=30,
                targetId=TARGET,
                informationCutoff=CUTOFF,
                outputContractId=CONTRACT,
                dataBundleSha256=request["data_bundle"]["sha256"],
                scientificPlanSha256=request["context"]["scientific_plan"]["sha256"],
                qualification=ready["summary"]["qualification"],
                strictAsOf=False,
                outputs={
                    "raceProbabilities": True,
                    "conditionalTwoPartyShares": True,
                    "chamberControl": False,
                    "tipping": False,
                    "counts": False,
                },
            )
        )
        examples.append(
            dict(
                id=f"ordinary-2018-{name.lower()}",
                contextId=context_id,
                activation="not_activated",
                experimental=True,
                sourceLifecycle="ready",
                numericalStatus="passed",
                evidenceStatus="unassessed",
                requestTemplate=template,
                requestSha256=ready["request_sha256"],
                configurationSha256=ready["record"]["configuration_sha256"],
                artifactId=ready["artifact"]["artifact_id"],
                readyFileSha256=sha(ready_file),
                summaryFileSha256=arm["summary"]["sha256"],
                quantileLevels=ready["summary"]["quantile_levels"],
                shareQuantileShape=[5, 30],
                limitations=ready["record"]["limitations"],
            )
        )
        lifecycle.append(
            dict(
                id=f"ordinary-2018-{name.lower()}-candidate-off",
                contextId=context_id,
                requestTemplate={**template, "selections": changed["selections"]},
                sourceLifecycle="requires_computation",
                displayedResultId=None,
                scheduled=False,
                reason=pending["reason"],
                sourceFingerprint=sha(boundary_path),
            )
        )
    require(
        len(examples) == 2 and {e["id"] for e in examples} == {"ordinary-2018-f", "ordinary-2018-fp"},
        "Expected both saved integration examples",
    )
    return dict(
        schemaVersion=2,
        id="senate-component-metadata-20261020-v2",
        activation="metadata_only",
        execution={
            "available": False,
            "schedulerAvailable": False,
            "reason": "This catalog describes reviewed research components. It does not run, queue or activate new forecasts.",
        },
        sourceFingerprints={
            "interfaceReceipt": INTERFACE_SHA256,
            "readyReceipt": READY_SHA256,
            "catalog": sha(output_paths["catalog.json"]),
            "resultIndex": sha(index_path),
            "schemas": {
                name.removesuffix(".json"): sha(path)
                for name, path in output_paths.items()
                if path.parent.name == "schemas"
            },
        },
        defaultPresetId=catalog["default_preset_id"],
        roles=roles,
        supportedContexts=contexts,
        savedExamples=examples,
        lifecycleExamples=lifecycle,
        limitations=[
            "Provider availability describes the bounded research backend, not execution availability on this site.",
            "These examples cover 30 ordinary Democratic–Republican Senate contests in 2018, with qualified retrospective inputs; no arbitrary 2026 composition is admitted here.",
            "Five roles have no providers in this catalog. Empty bindings do not establish a connected eight-layer scientific model.",
            "Ready, numerical checks passed, and predictive evidence are separate. The two examples remain empirically unassessed.",
            "Request templates omit server-owned source paths. A future trusted adapter must bind the exact data bundle and scientific plan before resolving a request.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/model-components-v2.json"))
    args = parser.parse_args()
    value = export_catalog(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Exported metadata for {len(value['roles'])} roles; no results activated")


if __name__ == "__main__":
    main()
