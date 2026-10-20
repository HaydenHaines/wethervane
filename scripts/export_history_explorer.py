#!/usr/bin/env python3
"""Publish read-only summaries of the sealed 2018/2022 chamber replays.

No forecast is fitted, resampled, or altered. Winner indices identify coalition
groups; a group containing several candidates is never split into invented
individual candidate probabilities.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EVIDENCE = Path("research/polling-layer-implementation-20261020")
INTAKE = EVIDENCE / "full-chamber-intake-sol"
CHAMBER = EVIDENCE / "full-chamber-root"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "public/data/explorer"
YEARS = (2018, 2022)
ARMS = ("FP", "F", "W", "FP_persistent")
LOADINGS = (1, 0, 0.5, 2)
CAUCUSES = ("D", "R", "U")
SCENARIOS = {
    "asof_alignment_other_U": (
        "As-of alignment; others unaligned",
        "Preserve the recorded pre-election organizing alignments; other groups remain unaligned.",
    ),
    "unresolved_other_D": (
        "Unaligned groups join D",
        "Counterfactual: groups otherwise unaligned join the Democratic caucus; existing alignments are preserved.",
    ),
    "unresolved_other_R": (
        "Unaligned groups join R",
        "Counterfactual: groups otherwise unaligned join the Republican caucus; existing alignments are preserved.",
    ),
}
ARM_LABELS = {
    "W": "Winner-only",
    "F": "Fundamentals",
    "FP": "Fundamentals + polls",
    "FP_persistent": "Fundamentals + persistent polling errors",
}


class HistoryExportError(ValueError):
    """A sealed input or accounting invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HistoryExportError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def checked_file(root: Path, relative: str | Path, expected_hash: str) -> Path:
    path = root / relative
    require(sha256(path) == expected_hash, f"Source hash mismatch: {relative}")
    return path


def sealed_file(directory: Path, seal: dict[str, Any], filename: str) -> Path:
    expected = seal.get("files", {}).get(filename)
    require(isinstance(expected, str), f"Unsealed file: {filename}")
    return checked_file(directory, filename, expected)


def distribution(seats: np.ndarray) -> dict[str, Any]:
    require(seats.ndim == 1 and len(seats) > 0, "Seat draws must be a nonempty vector")
    require(np.issubdtype(seats.dtype, np.integer), "Seat draws must be integers")
    require(bool(((seats >= 0) & (seats <= 100)).all()), "Seat draws outside chamber")
    pmf = np.bincount(seats, minlength=101) / len(seats)
    quantiles = np.quantile(seats, [0.05, 0.5, 0.95])
    return {
        "expectedSeats": float(seats.mean()),
        "seatQuantiles": dict(zip(("p05", "p50", "p95"), map(float, quantiles))),
        "pmf": pmf.tolist(),
    }


def control_probabilities(seats: np.ndarray, vice_president: str) -> dict[str, float]:
    require(vice_president in {"D", "R"}, "Unknown vice-presidential tie-break")
    thresholds = (50 if vice_president == "D" else 51, 50 if vice_president == "R" else 51)
    d = seats[:, 0] >= thresholds[0]
    r = seats[:, 1] >= thresholds[1]
    require(not bool((d & r).any()), "Both caucuses cannot control the chamber")
    return {
        "democraticMajorityProbability": float(d.mean()),
        "republicanMajorityProbability": float(r.mean()),
        "unresolvedOrganizingProbability": float((~d & ~r).mean()),
    }


def reconstruct_seats(
    winners: np.ndarray, contests: list[dict[str, Any]], scenario: dict[str, Any]
) -> np.ndarray:
    """Independently reconstruct every saved chamber row from group membership."""
    fixed = Counter(caucus for _, caucus in scenario["fixed_seats"])
    require(sum(fixed.values()) == 65, "Historical retained-seat ledger must contain 65 seats")
    result = np.tile([fixed[p] for p in CAUCUSES], (len(winners), 1))
    memberships = dict(scenario["race_memberships"])
    for column, contest in enumerate(contests):
        member_labels = memberships[contest["race_id"]]
        require(len(member_labels) == len(contest["group_ids"]), "Group/caucus membership mismatch")
        codes = np.array([CAUCUSES.index(p) for p in member_labels], dtype=np.int8)
        winning_caucus = codes[winners[:, column]]
        for caucus_index in range(3):
            result[:, caucus_index] += winning_caucus == caucus_index
    require(bool((result.sum(axis=1) == 100).all()), "Reconstructed chamber does not total 100")
    return result


def export_outcomes(
    contest: dict[str, Any], winner_column: np.ndarray,
    candidate_rows: dict[tuple[str, str], dict[str, str]],
    group_rows: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    race_id = contest["race_id"]
    group_ids = contest["group_ids"]
    require(bool(((winner_column >= 0) & (winner_column < len(group_ids))).all()), f"Invalid winner index: {race_id}")
    counts = np.bincount(winner_column, minlength=len(group_ids))
    outcomes = []
    for index, group_id in enumerate(group_ids):
        group = group_rows[(race_id, group_id)]
        members = []
        for candidate_id in group["candidate_ids"].split("|"):
            candidate = candidate_rows[(race_id, candidate_id)]
            require(candidate["group_id"] == group_id, f"Candidate/group identity mismatch: {race_id}")
            members.append({
                "id": candidate_id,
                "name": candidate["candidate_name"],
                "party": candidate["ballot_party"],
                "incumbent": candidate["incumbent"] == "True",
                "sourceUrls": [],
            })
        require(bool(members), f"Empty outcome group: {race_id}/{group_id}")
        residual = any(
            candidate_rows[(race_id, member["id"])].get("winner_support_kind")
            == "unnamed_eligible_writein_residual_hypothesis"
            for member in members
        )
        single = len(members) == 1 and not residual
        name = (
            f"{members[0]['name']} (unnamed write-in residual)" if residual
            else members[0]["name"] if single
            else f"{'Democratic' if group_id == 'D' else 'Republican' if group_id == 'R' else group_id} candidates"
        )
        outcomes.append({
            "id": group_id,
            "name": name,
            "party": members[0]["party"] if single else group["bloc"],
            "kind": "candidate" if single else "group",
            "members": members,
            "winProbability": float(counts[index] / len(winner_column)),
        })
    require(abs(sum(row["winProbability"] for row in outcomes) - 1) < 1e-12, f"Winner probabilities do not sum to one: {race_id}")
    return outcomes


def performance(score: dict[str, Any], observed: dict[str, str]) -> dict[str, Any]:
    actual = score["historical_actual_control_score"]
    require(score["numerical_passed"] is True, "Attempt to publish failed numerical cell")
    require(actual["observed_control"] == observed["observed_control"], "Scoring truth changed with scenario")
    seat_scores = score["caucus_seat_scores"]
    for party in CAUCUSES:
        require(seat_scores[party]["observed"] == int(observed[f"observed_{party}"]), "Scenario changed observed chamber truth")
    return {
        "observedSeats": {p: int(observed[f"observed_{p}"]) for p in CAUCUSES},
        "observedControl": actual["observed_control"],
        "dSeatCrps": seat_scores["D"]["crps"],
        "rSeatCrps": seat_scores["R"]["crps"],
        "controlBrier": actual["categorical_brier"],
        "controlLogScore": None if actual["negative_log_score_is_infinite"] else actual["negative_log_score"],
        "intervals": [
            {"party": p, **{key: interval[key] for key in ("level", "lower", "upper", "covered")}}
            for p in CAUCUSES for interval in seat_scores[p]["intervals"]
        ],
    }


def load_inputs(source_root: Path) -> dict[str, Any]:
    selection_path = source_root / CHAMBER / "selected-runs-v1.json"
    selection = read_json(selection_path)
    intake_dir = source_root / INTAKE
    freeze = read_json(intake_dir / "freeze-receipt-v1.json")
    freeze_pins = {row["path"]: row["sha256"] for row in freeze["files_before_receipt"]}
    names = ("candidates.csv", "groups.csv", "races.csv", "official-outcomes/labels-all-targets-SEPARATE.csv", "official-outcomes/observed-chamber.csv")
    paths = {name: checked_file(intake_dir, name, freeze_pins[name]) for name in names}
    candidates = read_csv(paths["candidates.csv"])
    groups = read_csv(paths["groups.csv"])
    races = read_csv(paths["races.csv"])
    labels = read_csv(paths["official-outcomes/labels-all-targets-SEPARATE.csv"])
    observed = read_csv(paths["official-outcomes/observed-chamber.csv"])
    candidate_map = {(row["race_id"], row["candidate_id"]): row for row in candidates}
    group_map = {(row["race_id"], row["group_id"]): row for row in groups}
    require(len(candidate_map) == len(candidates), "Duplicate candidate identity")
    require(len(group_map) == len(groups), "Duplicate outcome group identity")
    return {
        "selection": selection,
        "candidates": candidate_map,
        "groups": group_map,
        "races": {row["race_id"]: row for row in races},
        "labels": {row["race_id"]: row for row in labels},
        "observed": {int(row["cycle"]): row for row in observed},
        "sources": {
            "historicalRunSelectionSha256": sha256(selection_path),
            "historicalSupportFreezeSha256": sha256(intake_dir / "freeze-receipt-v1.json"),
            **{name.replace("/", "_"): sha256(path) for name, path in paths.items()},
        },
    }


def build_year(source_root: Path, inputs: dict[str, Any], year: int):
    """Yield complete ExplorerBundle objects, one internally coherent saved run each."""
    selection = next(row for row in inputs["selection"]["cycles"] if row["cycle"] == year)
    protocol_path = checked_file(source_root, selection["protocol"]["path"], selection["protocol"]["sha256"])
    protocol = read_json(protocol_path)
    require(protocol["primary_horizon_days"] == 14, "Unsupported historical horizon")
    base = source_root / selection["batch_path"] / str(year)
    stage_seals = {}
    for stage in ("fit", "core", "complete", "score"):
        reference = selection["stage_seals"][stage]
        stage_seals[stage] = read_json(checked_file(source_root, reference["path"], reference["sha256"]))
        require(stage_seals[stage]["protocol_sha256"] == selection["protocol"]["sha256"], "Mixed historical protocols")
    complete_seal = stage_seals["complete"]
    require(complete_seal["passed"] is True and complete_seal["target_outcome_values_parsed"] is False, "Prediction qualification failed")
    definition_path = sealed_file(base / "complete", complete_seal, "chamber-definition.json")
    definition = read_json(definition_path)
    contests = definition["contests"]
    race_ids = [row["race_id"] for row in contests]
    require(len(race_ids) == len(set(race_ids)) == 35, "Historical run must cover 35 distinct races")
    scores_path = sealed_file(base / "score", stage_seals["score"], "scores.json")
    scores = read_json(scores_path)
    require(len(scores) == 48 and stage_seals["score"]["cells_scored"] == 48, "Incomplete historical scores")
    score_map = {(row["arm"], float(row["loading"]), row["scenario"]): row for row in scores}
    require(len(score_map) == 48, "Duplicate historical score cell")
    observed = inputs["observed"][year]
    vp = observed["observed_vp_caucus"]
    thresholds = {"D": 50 if vp == "D" else 51, "R": 50 if vp == "R" else 51}
    cutoff_dates = {inputs["races"][race]["cutoff_date"] for race in race_ids}
    require(len(cutoff_dates) == 1, "Mixed historical cutoff dates")
    # The source records a day, not an authenticated intra-day issuance instant.
    cutoff = next(iter(cutoff_dates))
    common_sources = {
        **inputs["sources"],
        "historicalProtocolSha256": sha256(protocol_path),
        "historicalPredictionSealSha256": selection["stage_seals"]["complete"]["sha256"],
        "historicalScoreSealSha256": selection["stage_seals"]["score"]["sha256"],
        "historicalScoresSha256": sha256(scores_path),
        "historicalChamberDefinitionSha256": sha256(definition_path),
    }
    for arm in ARMS:
        for loading in LOADINGS:
            filename = f"{arm}-loading-{loading:g}.npz"
            source_path = sealed_file(base / "complete", complete_seal, filename)
            cell = next(row for row in complete_seal["cells"] if row["arm"] == arm and float(row["loading"]) == loading)
            require(cell["passed"] is True and cell["sha256"] == sha256(source_path), "Unqualified historical prediction cell")
            with np.load(source_path, allow_pickle=False) as draws:
                require(draws["race_ids"].tolist() == race_ids, "Winner draw race IDs differ from chamber definition")
                raw_winners = draws["winner_indices"]
                require(raw_winners.shape == (4, 20000, 35), "Unexpected joint historical draw dimensions")
                require(np.issubdtype(raw_winners.dtype, np.integer), "Winner indices must be integers")
                winners = raw_winners.reshape(-1, 35)
                races = []
                for column, contest in enumerate(contests):
                    race_id = contest["race_id"]
                    metadata = inputs["races"][race_id]
                    label = inputs["labels"][race_id]
                    outcomes = export_outcomes(contest, winners[:, column], inputs["candidates"], inputs["groups"])
                    require(label["winner_group_id"] in contest["group_ids"], "Observed winner outside forecast outcome support")
                    winner = inputs["candidates"][(race_id, label["ultimate_winner_candidate_id"])]
                    require(winner["group_id"] == label["winner_group_id"], "Observed winner candidate/group mismatch")
                    races.append({
                        "id": race_id,
                        "state": metadata["state"],
                        "seatId": contest["seat_id"],
                        "electionLabel": "Special election" if metadata["special"] == "True" else "Regular election",
                        "outcomes": outcomes,
                        "notes": [
                            "Probabilities describe modeled candidate groups. Multiple candidates in the same party group have no separate winner probabilities.",
                            "The incoming seat is defined after the final decisive voting event, including a required runoff; this is not an election-night forecast.",
                        ],
                        "observed": {
                            "outcomeId": label["winner_group_id"],
                            "winnerName": label["winner_name"],
                            "winnerParty": winner["ballot_party"],
                            "sourceUrls": [],
                        },
                    })
                scenarios = []
                for scenario in definition["scenarios"]:
                    scenario_id = scenario["name"]
                    score = score_map[(arm, float(loading), scenario_id)]
                    saved_seats = draws[f"seats:{scenario_id}"].reshape(-1, 3)
                    require(np.array_equal(saved_seats, reconstruct_seats(winners, contests, scenario)), "Saved chamber differs from group winner accounting")
                    controls = control_probabilities(saved_seats, vp)
                    actual_score = score["historical_actual_control_score"]
                    for short, name in (("D", "democraticMajorityProbability"), ("R", "republicanMajorityProbability"), ("U", "unresolvedOrganizingProbability")):
                        require(abs(controls[name] - actual_score["probabilities"][short]) < 1e-12, "Control summary differs from sealed score")
                    seat_distributions = {p: distribution(saved_seats[:, i]) for i, p in enumerate(CAUCUSES)}
                    for p in CAUCUSES:
                        scored = score["caucus_seat_scores"][p]
                        require(abs(seat_distributions[p]["expectedSeats"] - scored["mean"]) < 1e-10, "Seat mean differs from sealed score")
                        source_pmf = np.zeros(101)
                        for seat_count, probability in scored["probability_mass"].items():
                            source_pmf[int(seat_count)] = probability
                        require(np.allclose(seat_distributions[p]["pmf"], source_pmf, atol=1e-12, rtol=0), "Seat PMF differs from sealed score")
                    label, description = SCENARIOS[scenario_id]
                    fixed = Counter(p for _, p in scenario["fixed_seats"])
                    scenarios.append({
                        "id": scenario_id, "label": label, "description": description,
                        "thresholds": thresholds,
                        "retainedSeats": {p: fixed[p] for p in CAUCUSES},
                        "control": {"description": description, **controls},
                        "seats": seat_distributions,
                        "performance": performance(score, observed),
                    })
                run_id = f"history-{year}-{arm}-{loading:g}"
                uncertainty = f"loading-{loading:g}"
                yield {
                    "schemaVersion": 1,
                    "run": {
                        "id": run_id, "year": year,
                        "label": f"{year} · {ARM_LABELS[arm]} · loading {loading:g}",
                        "targetId": protocol["target_id"],
                        "lawId": "historical_categorical_winner_gate_v1",
                        "dateId": "h14", "dateLabel": "14 days before election",
                        "forecastAt": cutoff,
                        "issuedAt": inputs["selection"]["created_utc"],
                        "cutoffAt": cutoff, "horizonDays": 14,
                        "components": {
                            "base": "winner" if arm == "W" else "fundamentals",
                            "polling": "none" if arm in {"W", "F"} else "standard" if arm == "FP" else "persistent",
                            "uncertainty": uncertainty,
                        },
                        "uncertaintyLabel": f"Categorical national loading ×{loading:g}",
                        "uncertaintyDescription": "Changes the categorical winner component's shared national loading only; other shared parameters and ordinary-race dependence remain.",
                        "historical": True, "sensitivity": loading != 1,
                        "simulationCount": len(winners), "strictAsOf": False,
                    },
                    "races": races, "scenarios": scenarios,
                    "defaultScenarioId": "asof_alignment_other_U", "tipping": None,
                    "limitations": [
                        "Retrospective development benchmark; 2018 and 2022 were revisited during model development and are not blind holdouts.",
                        "Historical poll publication and candidate-support vintages use qualified retrospective proxies; this is not a strict real-time historical replay.",
                        "The historical categorical winner-gate law differs from the saved 2026 principal-pair forecast; these results do not directly calibrate that artifact.",
                        "Winner draws support group probabilities and chamber outcomes, not full-field vote-share margins or margin-ranked tipping frequencies.",
                        "Scattered/other unnamed write-in residuals are explicit source support hypotheses, not identified individual candidates or proof of exhaustive ballot eligibility.",
                        "Scores use fixed observed organizing alignment and the actual vice-presidential tie-break; changing a forecast caucus scenario does not change observed truth.",
                    ],
                    "historicalNote": "Saved 14-day retrospective replay, generated in 2026 with earlier-cycle parameter training. Historical cutoff and actual archive issuance are separate dates.",
                    "provenance": {"sources": {**common_sources, "historicalJointDrawsSha256": sha256(source_path)}},
                }


def write_json(path: Path, value: Any) -> str:
    content = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def export_history(source_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    inputs = load_inputs(source_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for year in YEARS:
        for bundle in build_year(source_root, inputs, year):
            filename = bundle["run"]["id"] + ".json"
            digest = write_json(output_dir / filename, bundle)
            entries.append({**bundle["run"], "href": "/data/explorer/" + filename, "sha256": digest})
    write_json(output_dir / "history-index.json", entries)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    entries = export_history(args.source_root, args.output_dir)
    print(f"Published {len(entries)} sealed historical run summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
