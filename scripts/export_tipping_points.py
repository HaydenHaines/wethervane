#!/usr/bin/env python3
"""Export caucus tipping-point summaries from saved joint forecast draws.

This script only reads and summarizes already-saved forecast artifacts. It does
not fit or resample either forecasting model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "public" / "data"
POLLING_DIR = Path("research/polling-layer-implementation-20261020/forecast-v2")
FUNDAMENTALS_DIR = Path("research/national-senate-forecast-20261020/forecast-v1")
EXPECTED_RACE_COUNT = 35
SCENARIO_FIELDS = ("democratic_caucus_members", "republican_caucus_members")
PARTY_THRESHOLDS = {"D": 51, "R": 50}


class TippingSnapshotError(ValueError):
    """Raised when saved artifacts fail the tipping snapshot contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TippingSnapshotError(f"Could not read {label} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TippingSnapshotError(f"{label} must be a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TippingSnapshotError(message)


def _scenario_map(source: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    scenarios = source.get("caucus_scenarios")
    if not isinstance(scenarios, list):
        raise TippingSnapshotError(f"{label} has no caucus_scenarios list")
    result: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        if not isinstance(scenario, dict) or not isinstance(scenario.get("id"), str):
            raise TippingSnapshotError(f"{label} has a malformed caucus scenario")
        scenario_id = scenario["id"]
        if scenario_id in result:
            raise TippingSnapshotError(f"{label} repeats scenario {scenario_id}")
        result[scenario_id] = scenario
    return result


def _validate_inputs_against_forecast(
    forecast: dict[str, Any], input_data: dict[str, Any], label: str
) -> list[dict[str, Any]]:
    races = forecast.get("races")
    input_races = input_data.get("races")
    if not isinstance(races, list) or not isinstance(input_races, list):
        raise TippingSnapshotError(f"{label} forecast and inputs must contain race lists")
    race_ids = [race.get("race_id") for race in races]
    input_ids = [race.get("race_id") for race in input_races]
    _require(
        len(races) == EXPECTED_RACE_COUNT
        and len(set(race_ids)) == EXPECTED_RACE_COUNT,
        f"{label} must contain {EXPECTED_RACE_COUNT} unique races",
    )
    _require(race_ids == input_ids, f"{label} forecast race order differs from its frozen inputs")
    for result_race, input_race in zip(races, input_races):
        for key in ("first_contender", "second_contender"):
            _require(
                result_race[key]["id"] == input_race[key]["id"],
                f"{label} contender order differs for {result_race['race_id']}",
            )

    forecast_scenarios = _scenario_map(forecast, label + " forecast")
    input_scenarios = _scenario_map(input_data, label + " inputs")
    _require(
        forecast_scenarios.keys() == input_scenarios.keys(),
        f"{label} forecast and inputs scenario IDs differ",
    )
    for scenario_id, scenario in forecast_scenarios.items():
        input_scenario = input_scenarios[scenario_id]
        for field in SCENARIO_FIELDS:
            _require(
                scenario.get(field) == input_scenario.get(field),
                f"{label} scenario {scenario_id} differs from frozen inputs in {field}",
            )
    _require(input_data.get("tie_break_party") == "R", f"{label} tie-break party is not Republican")
    return races


def _flatten_draws(shares: np.ndarray, first_wins: np.ndarray, race_count: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    if shares.ndim not in (2, 3) or shares.shape[-1] != race_count:
        raise TippingSnapshotError(f"{label} shares array has an unexpected shape: {shares.shape}")
    if first_wins.shape != shares.shape:
        raise TippingSnapshotError(f"{label} winner and share array shapes differ")
    shares = shares.reshape(-1, race_count)
    first_wins = first_wins.reshape(-1, race_count)
    if not np.isfinite(shares).all() or ((shares < 0) | (shares > 1)).any():
        raise TippingSnapshotError(f"{label} shares must be finite probabilities")
    if not np.array_equal(first_wins, shares > 0.5):
        raise TippingSnapshotError(f"{label} saved winner flags do not equal shares > 0.5")
    return shares, first_wins


def _validate_marginals(
    shares: np.ndarray, first_wins: np.ndarray, races: list[dict[str, Any]], label: str
) -> None:
    source_means = np.array([race["first_share_mean"] for race in races], dtype=float)
    source_win_probabilities = np.array(
        [race["first_win_probability"] for race in races], dtype=float
    )
    _require(
        np.allclose(shares.mean(axis=0), source_means, rtol=0.0, atol=1e-12),
        f"{label} draw share means do not match race summary order",
    )
    _require(
        np.allclose(first_wins.mean(axis=0), source_win_probabilities, rtol=0.0, atol=1e-12),
        f"{label} draw win probabilities do not match race summary order",
    )


def _chamber_groups(npz: Any, label: str) -> dict[str, np.ndarray]:
    if "chamber_group_names" in npz and "chamber_seats" in npz:
        names = npz["chamber_group_names"].tolist()
        seats = npz["chamber_seats"]
        if seats.ndim != 2 or seats.shape[1] != len(names):
            raise TippingSnapshotError(f"{label} chamber group matrix is malformed")
        return {str(name): seats[:, index] for index, name in enumerate(names)}
    return {str(key): npz[key] for key in npz.files}


def _member_configuration(
    races: list[dict[str, Any]],
    input_data: dict[str, Any],
    scenario: dict[str, Any],
    caucus: str,
) -> dict[str, Any]:
    field = "democratic_caucus_members" if caucus == "D" else "republican_caucus_members"
    members_list = scenario.get(field)
    if not isinstance(members_list, list) or not all(isinstance(x, str) for x in members_list):
        raise TippingSnapshotError(f"Scenario {scenario.get('id')} has malformed {field}")
    members = set(members_list)
    if len(members) != len(members_list):
        raise TippingSnapshotError(f"Scenario {scenario.get('id')} duplicates a {caucus} member")

    all_member_ids = {
        race[key]["id"]
        for race in races
        for key in ("first_contender", "second_contender")
    } | {seat["holder_id"] for seat in input_data["non_up_seats"]}
    _require(members <= all_member_ids, f"Scenario {scenario['id']} has unknown {caucus} member IDs")
    other_party = "R" if caucus == "D" else "D"
    other_field = "republican_caucus_members" if caucus == "D" else "democratic_caucus_members"
    other_members = set(scenario[other_field])
    _require(not members.intersection(other_members), f"Scenario {scenario['id']} overlaps D and R caucuses")

    retained_members = {
        seat["holder_id"]
        for seat in input_data["non_up_seats"]
        if seat["holder_id"] in members
    }
    _require(
        len(retained_members)
        == sum(1 for member in members if member.startswith("holder:")),
        f"Scenario {scenario['id']} has a retained member outside the 65-seat ledger",
    )
    guaranteed: list[int] = []
    eligible: list[int] = []
    member_candidate_by_index: dict[int, str] = {}
    for index, race in enumerate(races):
        first_id = race["first_contender"]["id"]
        second_id = race["second_contender"]["id"]
        first_member = first_id in members
        second_member = second_id in members
        if first_member and second_member:
            guaranteed.append(index)
        elif first_member or second_member:
            eligible.append(index)
            member_candidate_by_index[index] = first_id if first_member else second_id

    return {
        "members": members,
        "retained": len(retained_members),
        "guaranteed_indices": guaranteed,
        "eligible_indices": eligible,
        "member_candidate_by_index": member_candidate_by_index,
        "other_party": other_party,
    }


def _fractional_rank_frequencies(scores: np.ndarray, kth_from_top: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-draw kth scores and tie-fractional assignment weights."""
    draw_count, race_count = scores.shape
    if not 1 <= kth_from_top <= race_count:
        raise TippingSnapshotError("Requested order statistic is outside the eligible race set")
    kth_index = race_count - kth_from_top
    kth_score = np.partition(scores, kth_index, axis=1)[:, kth_index]
    equal = scores == kth_score[:, None]
    tie_count = equal.sum(axis=1)
    # The selected kth rank is one seat-order position. If a tie spans several
    # positions around the cutoff, split that one tipping-point assignment
    # evenly over the tied races so frequencies still sum to one per draw.
    weights = equal / tie_count[:, None]
    return kth_score, weights


def _summarize_caucus(
    *,
    party: str,
    scenario_id: str,
    configuration: dict[str, Any],
    races: list[dict[str, Any]],
    shares: np.ndarray,
    common_race_order: list[str],
    chamber_groups: dict[str, np.ndarray],
    source_control: dict[str, Any],
) -> dict[str, Any]:
    threshold = PARTY_THRESHOLDS[party]
    guaranteed_indices = configuration["guaranteed_indices"]
    eligible_indices = configuration["eligible_indices"]
    required = threshold - configuration["retained"] - len(guaranteed_indices)
    expected_contender_key = "first_contender" if party == "D" else "second_contender"
    _require(
        all(
            races[index][expected_contender_key]["id"]
            == configuration["member_candidate_by_index"][index]
            for index in eligible_indices
        ),
        f"{scenario_id}/{party} candidate orientation differs from frozen ribbon convention",
    )
    race_ids = [race["race_id"] for race in races]
    frequencies = {race_id: 0.0 for race_id in race_ids}
    status: str
    reason: str | None = None
    projected_race_id: str | None = None
    projected_margin_pp: float | None = None

    if required <= 0:
        status = "already_reached"
        reason = "retained_and_guaranteed_seats_meet_threshold"
    elif required > len(eligible_indices):
        status = "unreachable"
        reason = "required_contested_wins_exceed_eligible_races"
    else:
        status = "reachable"
        candidate_scores = np.column_stack(
            [
                shares[:, index]
                if races[index]["first_contender"]["id"]
                == configuration["member_candidate_by_index"][index]
                else 1.0 - shares[:, index]
                for index in eligible_indices
            ]
        )
        kth_score, assignment_weights = _fractional_rank_frequencies(candidate_scores, required)
        per_race_frequency = assignment_weights.mean(axis=0)
        for index, frequency in zip(eligible_indices, per_race_frequency):
            frequencies[race_ids[index]] = float(frequency)
        _require(
            abs(sum(frequencies.values()) - 1.0) <= 1e-12,
            f"{scenario_id}/{party} tipping frequencies do not sum to one",
        )

        # The projected marker uses the shared first-contender ribbon ordering,
        # reversed for Republican-caucus candidates (the second contender in
        # these frozen contest pairs). Member-share draws below still use IDs.
        eligible_ids = {race_ids[index] for index in eligible_indices}
        projected_order = [race_id for race_id in common_race_order if race_id in eligible_ids]
        if party == "R":
            projected_order = list(reversed(projected_order))
        projected_race_id = projected_order[required - 1]
        projected_index = race_ids.index(projected_race_id)
        candidate_id = configuration["member_candidate_by_index"][projected_index]
        first_is_member = races[projected_index]["first_contender"]["id"] == candidate_id
        member_mean_share = (
            float(shares[:, projected_index].mean())
            if first_is_member
            else float((1.0 - shares[:, projected_index]).mean())
        )
        projected_margin_pp = (2.0 * member_mean_share - 1.0) * 100.0

        seat_key = f"scenario:{scenario_id}:{party}"
        if seat_key not in chamber_groups:
            raise TippingSnapshotError(f"Saved chamber draws omit {seat_key}")
        expected_seats = (
            configuration["retained"]
            + len(guaranteed_indices)
            + (candidate_scores > 0.5).sum(axis=1)
        )
        source_seats = np.asarray(chamber_groups[seat_key])
        if source_seats.shape != expected_seats.shape or not np.array_equal(
            source_seats, expected_seats
        ):
            raise TippingSnapshotError(f"{scenario_id}/{party} race membership does not reconcile to chamber draws")
        tipping_control = kth_score > 0.5
        source_control_draws = source_seats >= threshold
        if not np.array_equal(tipping_control, source_control_draws):
            raise TippingSnapshotError(f"{scenario_id}/{party} tipping sign differs from chamber control draws")
        expected_control_probability = float(source_control_draws.mean())
        control_label = "democratic_majority_probability" if party == "D" else "republican_majority_probability"
        declared_control_probability = float(source_control[control_label])
        if not math.isclose(
            expected_control_probability, declared_control_probability, rel_tol=0.0, abs_tol=1e-12
        ):
            raise TippingSnapshotError(f"{scenario_id}/{party} chamber control probability does not reconcile")

    return {
        "thresholdSeats": threshold,
        "retainedSeats": configuration["retained"],
        "guaranteedRaceIds": [race_ids[index] for index in guaranteed_indices],
        "eligibleRaceIds": [race_ids[index] for index in eligible_indices],
        "eligibleCandidateByRace": {
            race_ids[index]: configuration["member_candidate_by_index"][index]
            for index in eligible_indices
        },
        "requiredContestedWins": required,
        "status": status,
        "statusReason": reason,
        "projectedTippingRaceId": projected_race_id,
        "projectedMarginPp": projected_margin_pp,
        "frequencyByRace": frequencies,
    }


def _model_snapshot(
    *,
    label: str,
    forecast: dict[str, Any],
    input_data: dict[str, Any],
    shares: np.ndarray,
    first_wins: np.ndarray,
    chamber_groups: dict[str, np.ndarray],
) -> dict[str, Any]:
    races = _validate_inputs_against_forecast(forecast, input_data, label)
    _validate_marginals(shares, first_wins, races, label)
    race_ids = [race["race_id"] for race in races]
    means = np.array([race["first_share_mean"] for race in races], dtype=float)
    common_order_indices = sorted(range(len(races)), key=lambda i: (-means[i], race_ids[i]))
    common_race_order = [race_ids[i] for i in common_order_indices]
    scenarios = _scenario_map(forecast, label + " forecast")
    controls = forecast.get("control_scenarios")
    if not isinstance(controls, dict) or controls.keys() != scenarios.keys():
        raise TippingSnapshotError(f"{label} control scenarios do not match caucus scenarios")

    output_scenarios: dict[str, dict[str, Any]] = {}
    for scenario_id, scenario in scenarios.items():
        caucuses = {}
        for party in ("D", "R"):
            configuration = _member_configuration(races, input_data, scenario, party)
            caucuses[party] = _summarize_caucus(
                party=party,
                scenario_id=scenario_id,
                configuration=configuration,
                races=races,
                shares=shares,
                common_race_order=common_race_order,
                chamber_groups=chamber_groups,
                source_control=controls[scenario_id],
            )
        output_scenarios[scenario_id] = caucuses
    return {
        "simulationCount": int(shares.shape[0]),
        "raceOrder": common_race_order,
        "scenarios": output_scenarios,
    }


def build_tipping_snapshot(
    source_root: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    polling_dir = source_root / POLLING_DIR
    fundamentals_dir = source_root / FUNDAMENTALS_DIR
    source_paths = {
        "pollingForecastSha256": polling_dir / "forecast.json",
        "pollingJointDrawsSha256": polling_dir / "joint-draws.npz",
        "fundamentalsForecastSha256": fundamentals_dir / "forecast.json",
        "fundamentalsJointDrawsSha256": fundamentals_dir / "joint-draws.npz",
    }
    source_hashes = {key: _sha256(path) for key, path in source_paths.items()}
    audit_artifacts = {
        "pollingFrozenInputsSha256": _sha256(
            polling_dir / "source-snapshots" / "10-inputs-frozen-root-v1.json"
        ),
        "fundamentalsFrozenInputsSha256": _sha256(fundamentals_dir / "inputs.json"),
        "pollingChamberDrawsSha256": _sha256(polling_dir / "chamber-draws.npz"),
    }

    polling_forecast = _read_json(polling_dir / "forecast.json", "polling forecast")
    polling_inputs = _read_json(
        polling_dir / "source-snapshots" / "10-inputs-frozen-root-v1.json", "polling frozen inputs"
    )
    fundamentals_forecast = _read_json(fundamentals_dir / "forecast.json", "fundamentals forecast")
    fundamentals_inputs = _read_json(fundamentals_dir / "inputs.json", "fundamentals frozen inputs")
    polling_npz_path = polling_dir / "joint-draws.npz"
    polling_chamber_path = polling_dir / "chamber-draws.npz"
    fundamentals_npz_path = fundamentals_dir / "joint-draws.npz"

    with np.load(polling_npz_path, allow_pickle=False) as polling_npz:
        polling_forecast_races = polling_forecast.get("races", [])
        polling_race_ids = [race.get("race_id") for race in polling_forecast_races]
        _require(len(polling_race_ids) == EXPECTED_RACE_COUNT, "Polling forecast race count changed")
        polling_shares, polling_wins = _flatten_draws(
            polling_npz["shares"], polling_npz["first_wins"], len(polling_race_ids), "polling"
        )
        _require(
            polling_shares.shape[0] == 40000,
            "Polling draw count changed from the frozen 4 × 10,000 simulation",
        )
        polling_chains = polling_npz["shares"].shape[0] if polling_npz["shares"].ndim == 3 else 1
        _require(polling_chains == 4, "Polling saved draws do not contain four chains")
    with np.load(polling_chamber_path, allow_pickle=False) as chamber_npz:
        polling_chamber = _chamber_groups(chamber_npz, "polling")

    with np.load(fundamentals_npz_path, allow_pickle=False) as fundamentals_npz:
        fund_json_ids = [race["race_id"] for race in fundamentals_forecast.get("races", [])]
        if "race_ids" not in fundamentals_npz:
            raise TippingSnapshotError("Fundamentals draws do not contain race_ids")
        fund_draw_ids = fundamentals_npz["race_ids"].tolist()
        _require(fund_draw_ids == fund_json_ids, "Fundamentals draw race IDs differ from forecast order")
        fundamentals_shares, fundamentals_wins = _flatten_draws(
            fundamentals_npz["shares"], fundamentals_npz["first_wins"], len(fund_json_ids), "fundamentals"
        )
        _require(
            fundamentals_shares.shape[0] == 100000,
            "Fundamentals draw count changed from the frozen 100,000 simulation",
        )
        fundamentals_chamber = _chamber_groups(fundamentals_npz, "fundamentals")

    polling_model = _model_snapshot(
        label="polling",
        forecast=polling_forecast,
        input_data=polling_inputs,
        shares=polling_shares,
        first_wins=polling_wins,
        chamber_groups=polling_chamber,
    )
    fundamentals_model = _model_snapshot(
        label="fundamentals",
        forecast=fundamentals_forecast,
        input_data=fundamentals_inputs,
        shares=fundamentals_shares,
        first_wins=fundamentals_wins,
        chamber_groups=fundamentals_chamber,
    )

    polling_issued = polling_forecast.get("execution_completed_utc")
    polling_cutoff = polling_forecast.get("input_cutoff_utc")
    if not isinstance(polling_issued, str) or not isinstance(polling_cutoff, str):
        raise TippingSnapshotError("Polling forecast is missing issuance or cutoff timestamps")
    snapshot = {
        "schemaVersion": 1,
        "method": "caucus_rank_threshold_v1",
        "issuedAt": polling_issued,
        "cutoffAt": polling_cutoff,
        "sources": source_hashes,
        "models": {"polling": polling_model, "fundamentals": fundamentals_model},
    }
    return snapshot, source_hashes, audit_artifacts


def export_tipping_snapshot(source_root: Path, output_dir: Path) -> tuple[Path, Path]:
    snapshot, source_hashes, audit_artifacts = build_tipping_snapshot(source_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    tipping_path = output_dir / "tipping-points.json"
    provenance_path = output_dir / "tipping-points-provenance.json"
    snapshot_bytes = (json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    tipping_path.write_bytes(snapshot_bytes)
    provenance = {
        "schemaVersion": 1,
        "snapshotSha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "sources": source_hashes,
        "auditArtifacts": audit_artifacts,
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return tipping_path, provenance_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    tipping_path, provenance_path = export_tipping_snapshot(args.source_root, args.output_dir)
    print(f"Wrote {tipping_path}")
    print(f"Wrote {provenance_path}")


if __name__ == "__main__":
    main()
