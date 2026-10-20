#!/usr/bin/env python3
"""Build static current Senate explorer bundles from saved forecast draws.

This adapter only transforms frozen JSON/NPZ/config artifacts. It never runs
forecast fits or changes source research artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.export_site_snapshot import (
        SnapshotError,
        _model as export_site_model,
    )
    from scripts.export_tipping_points import (
        EXPECTED_RACE_COUNT,
        _chamber_groups,
        _flatten_draws,
        _model_snapshot as summarize_tipping,
    )
else:
    from export_site_snapshot import (
        SnapshotError,
        _model as export_site_model,
    )
    from export_tipping_points import (
        EXPECTED_RACE_COUNT,
        _chamber_groups,
        _flatten_draws,
        _model_snapshot as summarize_tipping,
    )

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "public" / "data" / "explorer"

MODEL_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "id": "current-fundamentals",
        "label": "Fundamentals",
        "sourceKey": "fundamentals",
        "relativeDir": Path("research/national-senate-forecast-20261020/forecast-v1"),
        "inputRelativePath": Path("inputs.json"),
        "configurationRelativePath": Path("receipt.json"),
        "family": "fundamentals",
        "polling": "none",
        "uncertainty": "standard",
        "uncertaintyLabel": "Saved fundamentals posterior",
        "uncertaintyDescription": (
            "Uses the saved Senate fundamentals posterior and the October 20 race roster."
        ),
        "displayLabel": "October 20, 2026",
    },
    {
        "id": "current-polling",
        "label": "Standard polling",
        "sourceKey": "polling",
        "relativeDir": Path("research/polling-layer-implementation-20261020/forecast-v2"),
        "inputRelativePath": Path("source-snapshots/10-inputs-frozen-root-v1.json"),
        "configurationRelativePath": Path("run-specification.json"),
        "family": "polling",
        "polling": "standard",
        "uncertainty": "standard",
        "uncertaintyLabel": "Standard fitted polling-error scales",
        "uncertaintyDescription": (
            "Uses the saved standard covariance scales estimated from the historical polling calibration."
        ),
        "displayLabel": "October 20, 2026",
    },
    {
        "id": "current-error-low",
        "label": "Polling error: all scales × 2/3",
        "sourceKey": "polling",
        "relativeDir": Path("research/polling-layer-implementation-20261020/forecast-sensitivity-low-v1"),
        "inputRelativePath": Path("source-snapshots/10-inputs-frozen-root-v1.json"),
        "configurationRelativePath": Path("run-specification.json"),
        "family": "polling",
        "polling": "standard",
        "uncertainty": "error-low",
        "uncertaintyLabel": "All polling-error SDs × 2/3",
        "uncertaintyDescription": (
            "Sensitivity run scaling independent, cycle, race, pollster, and movement SDs by 2/3. "
            "This is an assumption scenario, not an uncertainty interval."
        ),
        "displayLabel": "October 20, 2026",
    },
    {
        "id": "current-error-high",
        "label": "Polling error: all scales × 1.5",
        "sourceKey": "polling",
        "relativeDir": Path("research/polling-layer-implementation-20261020/forecast-sensitivity-high-v1"),
        "inputRelativePath": Path("source-snapshots/10-inputs-frozen-root-v1.json"),
        "configurationRelativePath": Path("run-specification.json"),
        "family": "polling",
        "polling": "standard",
        "uncertainty": "error-high",
        "uncertaintyLabel": "All polling-error SDs × 1.5",
        "uncertaintyDescription": (
            "Sensitivity run scaling independent, cycle, race, pollster, and movement SDs by 1.5. "
            "This is an assumption scenario, not an uncertainty interval."
        ),
        "displayLabel": "October 20, 2026",
    },
    {
        "id": "current-cycle-low",
        "label": "Polling error: common-cycle SD × 0.5",
        "sourceKey": "polling",
        "relativeDir": Path("research/polling-layer-implementation-20261020/forecast-sensitivity-cycle-low-v1"),
        "inputRelativePath": Path("source-snapshots/10-inputs-frozen-root-v1.json"),
        "configurationRelativePath": Path("run-specification.json"),
        "family": "polling",
        "polling": "standard",
        "uncertainty": "cycle-low",
        "uncertaintyLabel": "Common-cycle error SD × 0.5",
        "uncertaintyDescription": (
            "Sensitivity run scaling only the shared cycle-error SD by 0.5. "
            "This is an assumption scenario, not an uncertainty interval."
        ),
        "displayLabel": "October 20, 2026",
    },
    {
        "id": "current-cycle-high",
        "label": "Polling error: common-cycle SD × 2",
        "sourceKey": "polling",
        "relativeDir": Path("research/polling-layer-implementation-20261020/forecast-sensitivity-cycle-high-v1"),
        "inputRelativePath": Path("source-snapshots/10-inputs-frozen-root-v1.json"),
        "configurationRelativePath": Path("run-specification.json"),
        "family": "polling",
        "polling": "standard",
        "uncertainty": "cycle-high",
        "uncertaintyLabel": "Common-cycle error SD × 2",
        "uncertaintyDescription": (
            "Sensitivity run scaling only the shared cycle-error SD by 2. "
            "This is an assumption scenario, not an uncertainty interval."
        ),
        "displayLabel": "October 20, 2026",
    },
)

SCENARIO_LABELS = {
    "campaign_pledge_unaligned": "Independents unaligned",
    "independent_opposition_caucuses_D": "Independents caucus with Democrats",
    "independent_opposition_caucuses_R": "Independents caucus with Republicans",
}

POLLING_SCALE_BY_UNCERTAINTY = {
    "standard": (1.0, 1.0),
    "error-low": (2.0 / 3.0, 1.0),
    "error-high": (1.5, 1.0),
    "cycle-low": (1.0, 0.5),
    "cycle-high": (1.0, 2.0),
}


class CurrentExplorerError(ValueError):
    """Raised when a saved current forecast fails the explorer artifact contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CurrentExplorerError(message)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CurrentExplorerError(f"Could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CurrentExplorerError(f"{label} must be a JSON object")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CurrentExplorerError(f"{label} must be a finite number")
    return float(value)


def _validate_polling_scales(spec: dict[str, Any], config: dict[str, Any]) -> None:
    expected = POLLING_SCALE_BY_UNCERTAINTY.get(spec["uncertainty"])
    _require(expected is not None, f"Unknown polling uncertainty label: {spec['uncertainty']}")
    observed = (
        _finite_number(config.get("all_scale_multiplier"), "all_scale_multiplier"),
        _finite_number(config.get("cycle_scale_multiplier"), "cycle_scale_multiplier"),
    )
    _require(
        all(math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-12)
            for actual, target in zip(observed, expected, strict=True)),
        f"{spec['id']} polling multipliers do not match its uncertainty label",
    )


def _candidate_outcome(candidate: dict[str, Any], win: float, mean: float,
                       quantiles: dict[str, float], denominator: str) -> dict[str, Any]:
    return {
        "id": candidate["id"],
        "name": candidate["name"],
        "party": candidate["party"],
        "kind": "candidate",
        "members": [{
            "id": candidate["id"],
            "name": candidate["name"],
            "party": candidate["party"],
            "incumbent": candidate["incumbent"],
            "sourceUrls": candidate["sourceUrls"],
        }],
        "winProbability": win,
        "share": {"mean": mean, "quantiles": quantiles, "denominator": denominator},
    }


def _race_outcomes(race: dict[str, Any], source_race: dict[str, Any]) -> dict[str, Any]:
    first, second = race["candidates"]
    q = race["shareQuantiles"]
    first_mean = race["firstShareMean"]
    second_quantiles = {"p05": 1.0 - q["p95"], "p50": 1.0 - q["p50"], "p95": 1.0 - q["p05"]}
    denominator = "Share of votes for the two named principal contenders"
    return {
        "id": race["id"],
        "state": race["state"],
        "seatId": race["seatId"],
        "electionLabel": "Special election" if race["id"].endswith(":special") else "Regular election",
        "outcomes": [
            _candidate_outcome(first, race["firstWinProbability"], first_mean, q, denominator),
            _candidate_outcome(
                second, race["secondWinProbability"], 1.0 - first_mean, second_quantiles, denominator
            ),
        ],
        **({"pollCount": race["pollCount"]} if "pollCount" in race else {}),
        "notes": race["notes"],
    }


def _summary_from_counts(counts: np.ndarray) -> dict[str, Any]:
    if counts.ndim != 1 or counts.size == 0:
        raise CurrentExplorerError("Caucus seat draw vector must be nonempty and one-dimensional")
    if not np.isfinite(counts).all() or not np.equal(counts, np.floor(counts)).all():
        raise CurrentExplorerError("Caucus seat draws must be finite integers")
    if ((counts < 0) | (counts > 100)).any():
        raise CurrentExplorerError("Caucus seat draws must lie within the 100-seat chamber")
    values = counts.astype(int)
    pmf = np.bincount(values, minlength=101).astype(float) / len(values)
    if not math.isclose(float(pmf.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise CurrentExplorerError("Derived seat PMF must sum to one")
    quantiles = np.quantile(values, [0.05, 0.50, 0.95])
    return {
        "expectedSeats": float(values.mean()),
        "seatQuantiles": {"p05": float(quantiles[0]), "p50": float(quantiles[1]), "p95": float(quantiles[2])},
        "pmf": pmf.tolist(),
    }


def _validate_saved_seat_summary(model: dict[str, Any], scenario_id: str, party: str,
                                 draws: np.ndarray) -> None:
    expected = _summary_from_counts(np.asarray(draws))
    declared = model["chamberDistributions"]["caucus"][scenario_id][party]
    _require(
        len(declared["pmf"]) == len(expected["pmf"])
        and np.allclose(declared["pmf"], expected["pmf"], rtol=0.0, atol=1e-12),
        f"Saved {party} chamber PMF does not match joint draws for {scenario_id}",
    )
    _require(
        math.isclose(declared["expectedSeats"], expected["expectedSeats"], rel_tol=0.0, abs_tol=1e-10),
        f"Saved {party} expected seats do not match joint draws for {scenario_id}",
    )
    for quantile in ("p05", "p50", "p95"):
        _require(
            math.isclose(declared["seatQuantiles"][quantile], expected["seatQuantiles"][quantile],
                         rel_tol=0.0, abs_tol=1e-10),
            f"Saved {party} {quantile} seats do not match joint draws for {scenario_id}",
        )


def _scenario_payload(
    model: dict[str, Any],
    tipping: dict[str, Any],
    source_scenario: dict[str, Any],
    draw_groups: dict[str, np.ndarray],
) -> dict[str, Any]:
    scenario_id = source_scenario["id"]
    scenario_tipping = tipping["scenarios"][scenario_id]
    d_key, r_key = f"scenario:{scenario_id}:D", f"scenario:{scenario_id}:R"
    _require(d_key in draw_groups and r_key in draw_groups, f"Missing saved caucus draw groups for {scenario_id}")
    d_draws, r_draws = np.asarray(draw_groups[d_key]), np.asarray(draw_groups[r_key])
    _require(d_draws.shape == r_draws.shape and d_draws.ndim == 1, f"Unaligned caucus draws for {scenario_id}")
    u_draws = 100 - d_draws - r_draws
    _validate_saved_seat_summary(model, scenario_id, "D", d_draws)
    _validate_saved_seat_summary(model, scenario_id, "R", r_draws)
    _require((u_draws >= 0).all(), f"D and R caucus seats exceed 100 in {scenario_id}")
    _require(
        np.array_equal(d_draws + r_draws + u_draws, np.full(d_draws.shape, 100)),
        f"Caucus seat draws do not reconcile to the 100-seat chamber in {scenario_id}",
    )
    source_caucus = model["chamberDistributions"]["caucus"][scenario_id]
    seats = {
        "D": source_caucus["D"],
        "R": source_caucus["R"],
        "U": _summary_from_counts(u_draws),
    }
    _require(
        len(seats["U"]["pmf"]) == 101
        and abs(sum(seats["U"]["pmf"]) - 1.0) <= 1e-12,
        f"Invalid derived U caucus PMF for {scenario_id}",
    )
    retained_d = scenario_tipping["D"]["retainedSeats"]
    retained_r = scenario_tipping["R"]["retainedSeats"]
    _require(retained_d == 34 and retained_r == 31, f"Unexpected retained-seat ledger for {scenario_id}")
    control = model["controls"][scenario_id]
    return {
        "id": scenario_id,
        "label": SCENARIO_LABELS.get(scenario_id, scenario_id),
        "description": control["description"],
        "thresholds": {"D": 51, "R": 50},
        "retainedSeats": {"D": retained_d, "R": retained_r, "U": 0},
        "control": control,
        "seats": seats,
    }


def _load_model_bundle(source_root: Path, spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    model_dir = source_root / spec["relativeDir"]
    forecast_path = model_dir / "forecast.json"
    joint_path = model_dir / "joint-draws.npz"
    config_path = model_dir / spec["configurationRelativePath"]
    inputs_path = model_dir / spec["inputRelativePath"]
    forecast = _read_object(forecast_path, f"{spec['id']} forecast")
    inputs = _read_object(inputs_path, f"{spec['id']} frozen inputs")
    config = _read_object(config_path, f"{spec['id']} run configuration")

    model_key = spec["sourceKey"]
    try:
        model = export_site_model(forecast, model_key)
    except SnapshotError as exc:
        raise CurrentExplorerError(f"{spec['id']} forecast failed public projection: {exc}") from exc

    with np.load(joint_path, allow_pickle=False) as joint:
        forecast_races = forecast.get("races", [])
        race_ids = [race.get("race_id") for race in forecast_races]
        _require(len(race_ids) == EXPECTED_RACE_COUNT, f"{spec['id']} expected 35 races")
        if "race_ids" in joint:
            _require(joint["race_ids"].tolist() == race_ids, f"{spec['id']} saved draw order differs from forecast")
        shares, wins = _flatten_draws(joint["shares"], joint["first_wins"], len(race_ids), spec["id"])
        if spec["family"] == "fundamentals":
            chamber_groups = _chamber_groups(joint, spec["id"])
        else:
            chamber_path = model_dir / "chamber-draws.npz"
            with np.load(chamber_path, allow_pickle=False) as chamber:
                chamber_groups = _chamber_groups(chamber, spec["id"])
        tipping = summarize_tipping(
            label=model_key,
            forecast=forecast,
            input_data=inputs,
            shares=shares,
            first_wins=wins,
            chamber_groups=chamber_groups,
        )

    scenarios = [
        _scenario_payload(model, tipping, row, chamber_groups)
        for row in forecast["caucus_scenarios"]
    ]
    by_id = {race["id"]: race for race in model["races"]}
    source_race_by_id = {race["race_id"]: race for race in forecast["races"]}
    races = [_race_outcomes(by_id[race_id], source_race_by_id[race_id]) for race_id in by_id]
    _require(len(races) == EXPECTED_RACE_COUNT, f"{spec['id']} public races changed count")

    issued = model["issuedAt"]
    cutoff = model["cutoffAt"]
    if spec["family"] == "fundamentals":
        simulation_count = int(config["draws"])
    else:
        _validate_polling_scales(spec, config)
        simulation_count = int(shares.shape[0])
    _require(simulation_count == shares.shape[0], f"{spec['id']} simulation count differs from saved draws")

    run = {
        "id": spec["id"],
        "year": 2026,
        "label": spec["label"],
        "targetId": model["targetId"],
        "lawId": "senate_principal_pair_aggregate_v1",
        "dateId": "2026-10-20",
        "dateLabel": spec["displayLabel"],
        "forecastAt": cutoff,
        "issuedAt": issued,
        "cutoffAt": cutoff,
        "horizonDays": None,
        "components": {
            "base": "fundamentals",
            "polling": spec["polling"],
            "uncertainty": spec["uncertainty"],
        },
        "uncertaintyLabel": spec["uncertaintyLabel"],
        "uncertaintyDescription": spec["uncertaintyDescription"],
        "historical": False,
        "sensitivity": bool(spec["id"] not in {"current-fundamentals", "current-polling"}),
        "simulationCount": simulation_count,
        "strictAsOf": True,
    }

    source_hashes = {
        "forecastSha256": sha256_file(forecast_path),
        "jointDrawsSha256": sha256_file(joint_path),
        "runConfigurationSha256": sha256_file(config_path),
        "frozenInputsSha256": sha256_file(inputs_path),
    }
    if spec["family"] == "fundamentals":
        source_hashes["runReceiptSha256"] = source_hashes.pop("runConfigurationSha256")
        source_hashes["posteriorSha256"] = str(config.get("posterior_sha256", ""))
        source_hashes["runnerSha256"] = str(config.get("runner_sha256", ""))
        source_hashes["fundamentalsCodeSha256"] = str(config.get("baseline_code_sha256", ""))
    else:
        chamber_path = model_dir / "chamber-draws.npz"
        source_hashes["chamberDrawsSha256"] = sha256_file(chamber_path)
        calibration_path = model_dir / "source-snapshots/07-calibration.json"
        source_hashes["calibrationSha256"] = sha256_file(calibration_path)
        for key, filename in (
            ("runnerCodeSha256", "00-standard_runner.py"),
            ("pollingCodeSha256", "03-polling_v1.py"),
            ("covarianceCodeSha256", "04-poll_covariance_v1.py"),
        ):
            source_hashes[key] = sha256_file(model_dir / "source-snapshots" / filename)

    bundle = {
        "schemaVersion": 1,
        "run": run,
        "races": races,
        "scenarios": scenarios,
        "defaultScenarioId": "campaign_pledge_unaligned",
        "tipping": tipping,
        "limitations": model["limitations"],
        "historicalNote": None,
        "provenance": {"sources": source_hashes},
    }
    return bundle, source_hashes


def _check_current_model_alignment(bundles: dict[str, dict[str, Any]]) -> None:
    fundamental = bundles["current-fundamentals"]
    baseline = {race["id"]: race for race in fundamental["races"]}
    for model_id, bundle in bundles.items():
        _require(len(bundle["races"]) == len(baseline), f"{model_id} race roster does not align with fundamentals")
        for race in bundle["races"]:
            expected = baseline.get(race["id"])
            _require(expected is not None, f"{model_id} contains unknown race {race['id']}")
            _require((race["state"], race["seatId"]) == (expected["state"], expected["seatId"]),
                     f"{model_id} race state/seat differs for {race['id']}")
            _require([outcome["id"] for outcome in race["outcomes"]] == [outcome["id"] for outcome in expected["outcomes"]],
                     f"{model_id} candidate IDs differ for {race['id']}")


def build_current_bundles(source_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    entries = []
    for spec in MODEL_SOURCES:
        bundle, _ = _load_model_bundle(source_root, spec)
        bundles[spec["id"]] = bundle
    _check_current_model_alignment(bundles)

    for spec in MODEL_SOURCES:
        bundle = bundles[spec["id"]]
        data = (json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        entries.append({**bundle["run"], "href": f"/data/explorer/{spec['id']}.json", "sha256": hashlib.sha256(data).hexdigest()})
    return bundles, entries


def export_current_bundles(source_root: Path, output_dir: Path) -> list[Path]:
    bundles, entries = build_current_bundles(source_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for model_id, bundle in bundles.items():
        path = output_dir / f"{model_id}.json"
        path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        written.append(path)
    index_path = output_dir / "current-index.json"
    index_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    written.append(index_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for path in export_current_bundles(args.source_root, args.output_dir):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
