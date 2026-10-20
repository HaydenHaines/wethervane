#!/usr/bin/env python3
"""Export the frozen October 20 Senate forecasts for a static public site.

This is a presentation adapter only. It validates and selects fields from the
saved fundamentals and polling artifacts; it does not rerun or change either
forecast.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "public" / "data"
RACE_COUNT = 35
EXPECTED_TARGETS = {
    "polling": "national_senate_aggregate_poll_v1",
    "fundamentals": "national_senate_aggregate_baseline_v1",
}
SOURCE_FILES = {
    "polling": Path(
        "research/polling-layer-implementation-20261020/forecast-v2/forecast.json"
    ),
    "fundamentals": Path(
        "research/national-senate-forecast-20261020/forecast-v1/forecast.json"
    ),
}


class SnapshotError(ValueError):
    """Raised when an input forecast does not satisfy the public snapshot contract."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotError(f"{label} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise SnapshotError(f"{label} must be finite")
    return numeric


def _probability(value: Any, label: str) -> float:
    numeric = _finite_number(value, label)
    if numeric < 0.0 or numeric > 1.0:
        raise SnapshotError(f"{label} must be between zero and one")
    return numeric


def _distribution(source: dict[str, Any], label: str) -> dict[str, Any]:
    quantiles = source.get("seat_quantiles")
    pmf = source.get("pmf")
    if not isinstance(quantiles, dict) or not isinstance(pmf, list) or not pmf:
        raise SnapshotError(f"{label} is missing seat quantiles or PMF")
    normalized_pmf = [_probability(p, f"{label}.pmf[{i}]") for i, p in enumerate(pmf)]
    if abs(sum(normalized_pmf) - 1.0) > 1e-8:
        raise SnapshotError(f"{label}.pmf must sum to one")
    return {
        "expectedSeats": _finite_number(source.get("expected_seats"), f"{label}.expected_seats"),
        "seatQuantiles": {
            key: _finite_number(quantiles[key], f"{label}.seat_quantiles.{key}")
            for key in ("p05", "p50", "p95")
        },
        "pmf": normalized_pmf,
    }


def _candidate(source: dict[str, Any], label: str) -> dict[str, Any]:
    candidate_id = source.get("id")
    name = source.get("name")
    party = source.get("party")
    incumbent = source.get("incumbent")
    urls = source.get("source_support", {}).get("source_urls")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise SnapshotError(f"{label}.id must be a nonempty string")
    if not isinstance(name, str) or not name:
        raise SnapshotError(f"{label}.name must be a nonempty string")
    if party not in {"D", "R", "I"}:
        raise SnapshotError(f"{label}.party must be D, R, or I")
    if not isinstance(incumbent, bool):
        raise SnapshotError(f"{label}.incumbent must be boolean")
    if not isinstance(urls, list):
        raise SnapshotError(f"{label}.sourceUrls must be a list")
    public_urls = [url for url in urls if isinstance(url, str) and url]
    if any(not (url.startswith("https://") or url.startswith("http://")) for url in public_urls):
        raise SnapshotError(f"{label}.sourceUrls must contain public HTTP(S) URLs")
    # source_support also contains local paths and source-specific raw metadata;
    # only reviewed public links are copied into the browser payload.
    return {
        "id": candidate_id,
        "name": name,
        "party": party,
        "incumbent": incumbent,
        "sourceUrls": public_urls,
    }


def _limitations(source: Any) -> list[str]:
    if isinstance(source, list) and all(isinstance(item, str) for item in source):
        return list(source)
    if isinstance(source, dict) and all(isinstance(value, str) for value in source.values()):
        return [f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in source.items()]
    raise SnapshotError("limitations must be a list of strings or a string-valued mapping")


def _race(source: dict[str, Any], model_key: str) -> dict[str, Any]:
    race_id = source.get("race_id")
    state = source.get("state")
    seat_id = source.get("seat_id")
    if not all(isinstance(value, str) and value for value in (race_id, state, seat_id)):
        raise SnapshotError(f"{model_key} race is missing id, state, or seat id")
    first = source.get("first_contender")
    second = source.get("second_contender")
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise SnapshotError(f"{race_id} is missing a named principal contender")

    first_win = _probability(source.get("first_win_probability"), f"{race_id}.first_win_probability")
    second_win = _probability(source.get("second_win_probability"), f"{race_id}.second_win_probability")
    if abs(first_win + second_win - 1.0) > 1e-8:
        raise SnapshotError(f"{race_id} named-contender win probabilities must sum to one")

    quantiles = source.get("first_share_quantiles")
    if not isinstance(quantiles, dict):
        raise SnapshotError(f"{race_id} is missing first-contender share quantiles")
    share_quantiles = {
        key: _probability(quantiles[key], f"{race_id}.first_share_quantiles.{key}")
        for key in ("p05", "p50", "p95")
    }
    if not share_quantiles["p05"] <= share_quantiles["p50"] <= share_quantiles["p95"]:
        raise SnapshotError(f"{race_id} share quantiles are not ordered")

    notes = [source.get("rule_proxy"), source.get("excluded_options_assumption")]
    if not all(isinstance(note, str) and note for note in notes):
        raise SnapshotError(f"{race_id} is missing its rule or omitted-option notes")

    race = {
        "id": race_id,
        "state": state,
        "seatId": seat_id,
        "candidates": [
            _candidate(first, f"{race_id}.first_contender"),
            _candidate(second, f"{race_id}.second_contender"),
        ],
        "firstWinProbability": first_win,
        "secondWinProbability": second_win,
        "firstShareMean": _probability(source.get("first_share_mean"), f"{race_id}.first_share_mean"),
        "shareQuantiles": share_quantiles,
        "notes": notes,
    }
    if model_key == "polling":
        poll_count = source.get("admitted_poll_observations")
        if isinstance(poll_count, int) and not isinstance(poll_count, bool) and poll_count >= 0:
            race["pollCount"] = poll_count
    return race


def _check_races(races: list[dict[str, Any]], model_key: str) -> None:
    ids = [race["id"] for race in races]
    if len(ids) != RACE_COUNT or len(set(ids)) != RACE_COUNT:
        raise SnapshotError(f"{model_key} must contain {RACE_COUNT} unique races")


def _model(source: dict[str, Any], model_key: str) -> dict[str, Any]:
    expected_target = EXPECTED_TARGETS[model_key]
    target = source.get("target_id")
    if target != expected_target:
        raise SnapshotError(f"{model_key} target_id must be {expected_target}")
    source_races = source.get("races")
    if not isinstance(source_races, list):
        raise SnapshotError(f"{model_key}.races must be a list")
    races = [_race(race, model_key) for race in source_races]
    _check_races(races, model_key)

    controls = source.get("control_scenarios")
    scenario_rows = source.get("caucus_scenarios")
    chamber = source.get("chamber")
    if not isinstance(controls, dict) or not isinstance(scenario_rows, list) or not isinstance(chamber, dict):
        raise SnapshotError(f"{model_key} is missing controls, caucus scenario, or chamber data")
    descriptions = {
        row.get("id"): row.get("description")
        for row in scenario_rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    if set(controls) != set(descriptions):
        raise SnapshotError(f"{model_key} caucus scenarios and control scenarios do not match")

    adapted_controls: dict[str, dict[str, Any]] = {}
    adapted_caucus: dict[str, dict[str, Any]] = {}
    for scenario_id, control in controls.items():
        if not isinstance(control, dict):
            raise SnapshotError(f"{model_key} control scenario {scenario_id} is malformed")
        probabilities = {
            "democraticMajorityProbability": _probability(
                control.get("democratic_majority_probability"), f"{scenario_id}.democratic_majority_probability"
            ),
            "republicanMajorityProbability": _probability(
                control.get("republican_majority_probability"), f"{scenario_id}.republican_majority_probability"
            ),
            "unresolvedOrganizingProbability": _probability(
                control.get("unresolved_organizing_probability"), f"{scenario_id}.unresolved_organizing_probability"
            ),
        }
        if abs(sum(probabilities.values()) - 1.0) > 1e-8:
            raise SnapshotError(f"{scenario_id} control probabilities must sum to one")
        description = descriptions[scenario_id]
        if not isinstance(description, str) or not description:
            raise SnapshotError(f"{scenario_id} is missing a caucus description")
        adapted_controls[scenario_id] = {"description": description, **probabilities}
        adapted_caucus[scenario_id] = {}
        for party in ("D", "R"):
            key = f"scenario:{scenario_id}:{party}"
            if key not in chamber:
                raise SnapshotError(f"{model_key} is missing chamber distribution {key}")
            adapted_caucus[scenario_id][party] = _distribution(chamber[key], f"chamber.{key}")

    party_distributions: dict[str, dict[str, Any]] = {}
    for party in ("D", "R", "I"):
        key = f"party_{party}"
        if key not in chamber:
            raise SnapshotError(f"{model_key} is missing party chamber distribution {key}")
        party_distributions[party] = _distribution(chamber[key], f"chamber.{key}")

    issued_at = source.get("execution_completed_utc", source.get("issuance_utc"))
    cutoff_at = source.get("input_cutoff_utc")
    if not isinstance(issued_at, str) or not isinstance(cutoff_at, str):
        raise SnapshotError(f"{model_key} is missing issuance or cutoff timestamp")
    return {
        "targetId": target,
        "issuedAt": issued_at,
        "cutoffAt": cutoff_at,
        "races": races,
        "controls": adapted_controls,
        "chamberDistributions": {"party": party_distributions, "caucus": adapted_caucus},
        "limitations": _limitations(source.get("limitations")),
    }


def _validate_identity_alignment(polling: dict[str, Any], fundamentals: dict[str, Any]) -> None:
    polling_by_id = {race["id"]: race for race in polling["races"]}
    fundamentals_by_id = {race["id"]: race for race in fundamentals["races"]}
    if polling_by_id.keys() != fundamentals_by_id.keys():
        raise SnapshotError("polling and fundamentals race IDs do not align")
    for race_id, poll_race in polling_by_id.items():
        fundamental_race = fundamentals_by_id[race_id]
        if (poll_race["state"], poll_race["seatId"]) != (
            fundamental_race["state"], fundamental_race["seatId"]
        ):
            raise SnapshotError(f"{race_id} state or seat identity differs between forecasts")
        poll_candidates = poll_race["candidates"]
        fundamental_candidates = fundamental_race["candidates"]
        for poll_candidate, fundamental_candidate in zip(poll_candidates, fundamental_candidates):
            fields = ("id", "name", "party", "incumbent")
            if any(poll_candidate[field] != fundamental_candidate[field] for field in fields):
                raise SnapshotError(f"{race_id} candidate identity differs between forecasts")


def build_snapshot(source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    source_records: dict[str, dict[str, str]] = {}
    for model_key, relative_path in SOURCE_FILES.items():
        artifact_path = source_root / relative_path
        content = artifact_path.read_bytes()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise SnapshotError(f"{model_key} source artifact is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise SnapshotError(f"{model_key} source artifact must be a JSON object")
        loaded[model_key] = parsed
        source_records[model_key] = {
            "sha256": sha256_bytes(content),
            "targetId": str(parsed.get("target_id", "")),
            "artifactLabel": "2026 October 20 standard polling forecast"
            if model_key == "polling"
            else "2026 October 20 Senate fundamentals forecast",
        }

    polling = _model(loaded["polling"], "polling")
    fundamentals = _model(loaded["fundamentals"], "fundamentals")
    _validate_identity_alignment(polling, fundamentals)
    snapshot = {
        "schemaVersion": 1,
        "issuedAt": polling["issuedAt"],
        "cutoffAt": polling["cutoffAt"],
        "sources": {
            "pollingSha256": source_records["polling"]["sha256"],
            "fundamentalsSha256": source_records["fundamentals"]["sha256"],
        },
        "models": {"polling": polling, "fundamentals": fundamentals},
    }
    provenance = {
        "schemaVersion": 1,
        "sources": source_records,
    }
    return snapshot, provenance


def export_snapshot(source_root: Path, output_dir: Path) -> tuple[Path, Path]:
    snapshot, provenance = build_snapshot(source_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = output_dir / "forecast.json"
    provenance_path = output_dir / "provenance.json"
    forecast_bytes = (json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    forecast_path.write_bytes(forecast_bytes)
    provenance["snapshotSha256"] = sha256_bytes(forecast_bytes)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return forecast_path, provenance_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Research checkout containing the frozen forecast artifacts",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    forecast_path, provenance_path = export_snapshot(args.source_root, args.output_dir)
    print(f"Wrote {forecast_path}")
    print(f"Wrote {provenance_path}")


if __name__ == "__main__":
    main()
