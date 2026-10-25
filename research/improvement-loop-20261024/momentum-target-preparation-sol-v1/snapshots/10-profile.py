"""No-fit Senate poll coverage profile. Writes only to this new packet."""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
LAB = Path("/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024")
INTEG = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
BASE = INTEG / "research/polling-layer-implementation-20261020"
V2 = LAB / "momentum-audit-v2"
ARCHIVE = INTEG / ".redesign-runtime/forecast-20261007/polls/senate_polls_historical.csv"
HIST = BASE / "historical-intake-sol/derived/historical_poll_panel.csv"
ROSTER_A = BASE / "full-chamber-intake-sol/event-roster-2018-2022.csv"
ROSTER_B = BASE / "cross-context-intake-sol/roster-2020-2024.csv"
SELECTION = LAB / "checkpoints/momentum-preflight-root-selection-v1.json"
F0_PLAN = LAB / "economic-experiment-v1/prepared-v1/plan.json"
BANK_COMPAT = LAB / "momentum-preflight-bank-compatibility-sol-v1/compatibility.json"
HORIZONS = (30, 14, 7)
LAGS = (0, 1, 3, 7)
MAX_AGE = 100


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def ident(x):
    if pd.isna(x):
        return ""
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return str(x)


def pollster_key(value, sample):
    s = "" if pd.isna(value) else str(value).strip().casefold()
    if not s or s in {"nan", "none", "unknown", "<na>"}:
        return "__unknown_sample__:" + str(sample)
    return s


def overlap_count(f):
    overlap = total = 0
    if f.empty or not {"start_date", "end_date"}.issubset(f):
        return 0, 0
    for _, g in f.groupby(["race_id", "pollster_key"], sort=True):
        g = g.dropna(subset=["start_date", "end_date"]).sort_values(
            ["start_date", "end_date", "observation_id"]
        )
        rs = list(g.itertuples(index=False))
        for i, a in enumerate(rs):
            for b in rs[i + 1:]:
                total += 1
                if a.start_date <= b.end_date and b.start_date <= a.end_date:
                    overlap += 1
    return overlap, total


def metrics(f, bank):
    if f.empty:
        return dict(rows=0, unique_samples=0, races=0, pollsters_known=0,
                    repeated_race_pollster_groups=0, races_with_repeated_pollster=0,
                    repeated_group_rows=0, median_repeated_group_span_days=None,
                    median_race_time_span_days=None, interval_rows_known=0,
                    interval_overlap_pairs=0, interval_comparable_pairs=0,
                    bank_races_supported=0, bank_races=len(bank))
    spans, group_races, group_rows = [], set(), 0
    for (race, pk), g in f.groupby(["race_id", "pollster_key"], sort=True):
        d = pd.to_datetime(g.end_date).dropna().drop_duplicates()
        if len(d) >= 2:
            spans.append(float((d.max() - d.min()).days))
            group_races.add(race)
            group_rows += len(g)
    race_spans = []
    for _, g in f.groupby("race_id", sort=True):
        d = pd.to_datetime(g.end_date).dropna()
        if len(d) >= 2:
            race_spans.append(float((d.max() - d.min()).days))
    overlap, comparable = overlap_count(f)
    starts = pd.to_datetime(f.start_date, errors="coerce")
    ends = pd.to_datetime(f.end_date, errors="coerce")
    known = int((starts.notna() & ends.notna()).sum())
    return dict(
        rows=len(f), unique_samples=int(f.sample_id.nunique()), races=int(f.race_id.nunique()),
        pollsters_known=int(f.loc[~f.pollster_key.str.startswith("__unknown_sample__:"), "pollster_key"].nunique()),
        repeated_race_pollster_groups=len(spans), races_with_repeated_pollster=len(group_races),
        repeated_group_rows=group_rows,
        median_repeated_group_span_days=None if not spans else float(pd.Series(spans).median()),
        median_race_time_span_days=None if not race_spans else float(pd.Series(race_spans).median()),
        interval_rows_known=known, interval_overlap_pairs=overlap,
        interval_comparable_pairs=comparable,
        bank_races_supported=len(set(f.race_id.astype(str)) & bank), bank_races=len(bank),
    )


def main():
    source_scripts = {"profile.py", "historical_calibration_baseline.py", "interval_support.py"}
    require({p.name for p in OUT.iterdir()} <= source_scripts,
            "refusing to overwrite existing generated evidence; use a fresh sibling output directory")
    require(sha(SELECTION) == "5de976e5facbc84aa9db48e64fcc43864d1b8741fc5d1ddc134cc2e0fa05e5e9",
            "root selection hash mismatch")
    require(sha(V2 / "artifact-manifest.json") == "eef2616076d7e6070a47c73c2b18637a10bd3ff82a65d87617d27b9303efd872",
            "v2 manifest hash mismatch")
    manifest = json.loads((V2 / "artifact-manifest.json").read_text())
    for f in manifest["files"]:
        require(sha(V2 / f["path"]) == f["sha256"], "v2 artifact changed: " + f["path"])
    sys.path.insert(0, str(V2))
    import identity
    identity.authenticate()
    sys.path.insert(0, str(INTEG))
    from src.forecasting.senate_data import senate_polls_as_of

    f0 = json.loads(F0_PLAN.read_text())
    require(sha(F0_PLAN) == "cfe669f6cef8cec82702b03e9ced060edf022fd434df131483515a054d24aef9",
            "F0 support plan hash mismatch")
    bank_by_cycle = {int(x["target_cycle"]): set(x["target_race_ids"]) for x in f0["folds"]}
    require(sha(BANK_COMPAT) == "53337fcbd640c1ece9c31f4c2c44cf107bbdcd60bc5bee912716109a3806ba05",
            "Sol bank compatibility binding changed")
    compat = json.loads(BANK_COMPAT.read_text())
    f0_banks = {int(x["year"]): x for x in compat["banks"] if x["arm"] == "F0"}
    require(set(f0_banks) == set(bank_by_cycle) and
            all(set(f0_banks[y]["target_race_ids"]) == bank_by_cycle[y] for y in bank_by_cycle),
            "F0 support differs between prepared plan and reviewed-bank manifest")

    roster = pd.concat([pd.read_csv(ROSTER_A), pd.read_csv(ROSTER_B)], ignore_index=True)
    roster = roster[roster.cycle.isin([2018, 2020, 2022, 2024]) & roster.eligible.eq(True) & roster.special.eq(False)].copy()
    for c in ["source_race_id", "dem_candidate_id", "rep_candidate_id"]:
        roster[c] = roster[c].map(ident)
    roster["cycle"] = roster.cycle.astype(int)
    roster["election_date"] = pd.to_datetime(roster.election_date)
    require(not roster.race_id.duplicated().any(), "duplicate eligible target race IDs")

    raw = pd.read_csv(ARCHIVE, low_memory=False)
    raw["cycle"] = pd.to_numeric(raw.cycle, errors="coerce").astype("Int64")
    for c in ["created_at", "end_date", "start_date"]:
        raw[c] = pd.to_datetime(raw[c], format="mixed", errors="coerce")
    cells, selected_rows, exclusions, helper_audits, support, race_horizon = [], [], [], [], [], []
    for cycle in [2018, 2020, 2022, 2024]:
        rr = roster[roster.cycle.eq(cycle)].copy()
        election = pd.Timestamp(rr.election_date.iloc[0]).normalize()
        bank = bank_by_cycle[cycle]
        # Final pre-election native-helper set, used only to enumerate cutoff exclusions.
        src0 = raw[raw.cycle.eq(cycle)].copy()
        base_cutoff = election - pd.Timedelta(days=1)
        base_proxy = src0.created_at.dt.normalize() + pd.Timedelta(days=1)
        base_input = src0[src0.end_date.le(base_cutoff) & src0.created_at.notna() & base_proxy.le(base_cutoff)]
        base_selected, _ = senate_polls_as_of(base_input, rr, cycle=cycle, cutoff=base_cutoff)
        if len(base_selected):
            base_selected = base_selected.copy()
            base_selected["end_date"] = pd.to_datetime(base_selected.end_date, errors="coerce")
            base_selected["available_date"] = pd.to_datetime(base_selected.available_date, errors="coerce")
            base_selected["end_lead"] = (election - base_selected.end_date).dt.days
            base_selected = base_selected[base_selected.end_lead.between(0, MAX_AGE, inclusive="both")]
        for h in HORIZONS:
            cutoff = election - pd.Timedelta(days=h)
            for lag in LAGS:
                src = raw[raw.cycle.eq(cycle)].copy()
                native_proxy = src.created_at.dt.normalize() + pd.Timedelta(days=1)
                allowed = src.end_date.le(cutoff) & src.created_at.notna() & (native_proxy + pd.Timedelta(days=lag)).le(cutoff)
                filtered = src.loc[allowed]
                f, audit = senate_polls_as_of(filtered, rr, cycle=cycle, cutoff=cutoff)
                audit.update(cycle=cycle, horizon_days=h, extra_lag_days=lag,
                             cutoff_utc=cutoff.strftime("%Y-%m-%dT00:00:00Z"),
                             availability_proxy="created_at_date_plus_1_day_then_extra_lag",
                             strict_asof_qualified=False, input_rows_after_proxy_filter=int(len(filtered)))
                helper_audits.append(audit)
                if f.empty:
                    f = pd.DataFrame(columns=["observation_id", "sample_id", "poll_id", "question_id",
                                              "race_id", "pollster", "start_date", "end_date",
                                              "available_date", "cycle"])
                f = f.copy()
                for c in ["start_date", "end_date", "available_date"]:
                    f[c] = pd.to_datetime(f[c], errors="coerce")
                if len(f):
                    f["end_lead"] = (election - f.end_date).dt.days
                    f = f[f.end_lead.between(h, MAX_AGE, inclusive="both")].copy()
                f["pollster_key"] = [pollster_key(p, s) for p, s in zip(f.pollster, f.sample_id)]
                latest = f.sort_values(["end_date", "available_date", "sample_id", "observation_id"]).drop_duplicates(
                    ["race_id", "pollster_key"], keep="last"
                )
                latest_ids = set(latest.observation_id.astype(str))
                l = f[f.observation_id.astype(str).isin(latest_ids)].copy()
                for name, part in [("538_native_full_series", f), ("538_latest_per_race_pollster", l)]:
                    cells.append(dict(source_panel="538_target_archive", cycle=cycle, horizon_days=h,
                                      extra_lag_days_beyond_native_proxy=lag, operator=name, **metrics(part, bank)))
                roster_ids = set(rr.race_id.astype(str))
                for rid in sorted(bank | roster_ids):
                    native_r = f[f.race_id.astype(str).eq(rid)]
                    latest_r = l[l.race_id.astype(str).eq(rid)]
                    groups = native_r.groupby("pollster_key").filter(lambda x: x.end_date.nunique() >= 2) if len(native_r) else native_r
                    race_horizon.append(dict(
                        source_panel="538_target_archive", cycle=cycle, horizon_days=h,
                        extra_lag_days_beyond_native_proxy=lag, cutoff_utc=cutoff.strftime("%Y-%m-%dT00:00:00Z"),
                        race_id=rid, in_v2_eligible_roster=rid in roster_ids, in_f0_bank_target=rid in bank,
                        native_samples=int(native_r.sample_id.nunique()), latest_pollster_samples=int(latest_r.sample_id.nunique()),
                        native_pollster_count=int(native_r.pollster_key.nunique()),
                        repeated_same_pollster_groups=int(groups.pollster_key.nunique()) if len(groups) else 0,
                        native_observation_ids="|".join(sorted(native_r.observation_id.astype(str))),
                        latest_observation_ids="|".join(sorted(latest_r.observation_id.astype(str))),
                        disposition="polls_in_common_roster_bank" if rid in bank and rid in roster_ids and len(native_r) else
                                    "no_helper_admitted_sample_at_cutoff" if rid in bank and rid in roster_ids else
                                    "bank_only_without_v2_eligible_poll_roster" if rid in bank else "roster_race_not_in_f0_bank"))
                for r in f.itertuples(index=False):
                    base = dict(source_panel="538_target_archive", cycle=cycle, horizon_days=h,
                                extra_lag_days_beyond_native_proxy=lag, cutoff_utc=cutoff.strftime("%Y-%m-%dT00:00:00Z"),
                                observation_id=str(r.observation_id), sample_id=str(r.sample_id),
                                poll_id=str(r.poll_id), question_id=str(r.question_id), race_id=str(r.race_id),
                                pollster=str(r.pollster), pollster_key=str(r.pollster_key),
                                start_date="" if pd.isna(r.start_date) else pd.Timestamp(r.start_date).date().isoformat(),
                                end_date=pd.Timestamp(r.end_date).date().isoformat(),
                                available_date_proxy=pd.Timestamp(r.available_date).date().isoformat(),
                                end_lead_days=int(r.end_lead), in_f0_bank_target=str(r.race_id) in bank)
                    selected_rows.append(base | dict(operator="538_native_full_series", selected=True,
                                                     disposition="native_selected_observation"))
                    selected_rows.append(base | dict(operator="538_latest_per_race_pollster",
                                                     selected=str(r.observation_id) in latest_ids,
                                                     disposition="latest_for_race_pollster" if str(r.observation_id) in latest_ids
                                                     else "older_observation_for_same_race_pollster"))
                    if str(r.observation_id) not in latest_ids:
                        exclusions.append(base | dict(operator="538_latest_per_race_pollster",
                                                      exclusion_reason="older_observation_for_same_race_pollster"))
                current_ids = set(f.observation_id.astype(str))
                current_samples = set(f.sample_id.astype(str))
                for r in base_selected.itertuples(index=False):
                    oid = str(r.observation_id)
                    if oid in current_ids:
                        continue
                    end = pd.Timestamp(r.end_date)
                    proxy = pd.Timestamp(r.available_date)
                    if (election - end).days > MAX_AGE:
                        why = "outside_100_day_maximum"
                    elif end > cutoff:
                        why = "field_end_after_requested_horizon"
                    elif proxy + pd.Timedelta(days=lag) > cutoff:
                        why = "native_availability_proxy_plus_extra_lag_after_horizon"
                    elif str(r.sample_id) in current_samples:
                        why = "alternate_question_selected_after_cutoff_refilter"
                    else:
                        why = "not_selected_by_horizon_helper_or_question_gate"
                    exclusions.append(dict(source_panel="538_target_archive", cycle=cycle, horizon_days=h,
                                           extra_lag_days_beyond_native_proxy=lag,
                                           operator="538_native_full_series", observation_id=oid,
                                           sample_id=str(r.sample_id), poll_id=str(r.poll_id),
                                           question_id=str(r.question_id), race_id=str(r.race_id),
                                           exclusion_reason=why))

    hist = pd.read_csv(HIST, low_memory=False)
    hist["end_date"] = pd.to_datetime(hist.end_date, errors="coerce")
    hist["available_date"] = pd.to_datetime(hist.available_date, errors="coerce")
    hist["days_to_election"] = pd.to_numeric(hist.days_to_election, errors="coerce")
    hist["cycle"] = hist.cycle.astype(int)
    hist["pollster_key"] = [pollster_key(p, s) for p, s in zip(hist.pollster, hist.sample_id)]
    hist_cells = []
    hist_selection_rows = []
    hist_exclusions = []
    for target_cycle in [2018, 2020, 2022, 2024]:
        f0_bank = f0_banks[target_cycle]
        bank = set(f0_bank["training_race_ids"])
        training_cycles = [c for c in [2012, 2016, 2018, 2020] if c < target_cycle]
        source = hist[hist.cycle.isin(training_cycles)].copy()
        for h in HORIZONS:
            for lag in LAGS:
                # Field-end proxy plus extra lag; integer day leads make the cutoff explicit.
                elig = source[source.days_to_election.between(h + lag, MAX_AGE, inclusive="both")].copy()
                elig["lead_bin"] = elig.days_to_election.astype(int) // 14
                bins = elig.sort_values(["end_date", "observation_id"]).drop_duplicates(
                    ["cycle", "race_id", "pollster_key", "lead_bin"], keep="last"
                )
                for name, part in [("tierney_native_full_series", elig),
                                   ("tierney_historical_latest_per_14day_bin", bins)]:
                    hist_cells.append(dict(source_panel="tierney_historical_calibration", target_cycle=target_cycle,
                                           horizon_days=h, extra_lag_days_beyond_native_proxy=lag,
                                           training_cycles=",".join(map(str, training_cycles)),
                                           operator=name, **metrics(part, bank)))
                retained = set(bins.observation_id.astype(str))
                admitted = set(elig.observation_id.astype(str))
                for r in source.itertuples(index=False):
                    oid = str(r.observation_id)
                    lead = None if pd.isna(r.days_to_election) else float(r.days_to_election)
                    in_window = oid in admitted
                    hist_selection_rows.append(dict(
                        source_panel="tierney_historical_calibration", target_cycle=target_cycle,
                        cycle=int(r.cycle), training_cycles=",".join(map(str, training_cycles)), horizon_days=h,
                        extra_lag_days_beyond_native_proxy=lag, cutoff_utc="Election date minus horizon at 00:00 UTC",
                        operator="tierney_historical_latest_per_14day_bin", observation_id=oid,
                        sample_id=str(r.sample_id), poll_id="", question_id="", race_id=str(r.race_id),
                        pollster=str(r.pollster), pollster_key=str(r.pollster_key), start_date="",
                        end_date="" if pd.isna(r.end_date) else pd.Timestamp(r.end_date).date().isoformat(),
                        available_date_proxy="" if pd.isna(r.available_date) else pd.Timestamp(r.available_date).date().isoformat(),
                        end_lead_days=lead, selected=bool(in_window and oid in retained),
                        in_f0_training_bank=str(r.race_id) in bank,
                        disposition=("retained_latest_in_fixed_14day_bin" if in_window and oid in retained
                                     else "superseded_within_fixed_14day_bin" if in_window
                                     else "outside_horizon_max_age_or_extra_lag")))
                    if not in_window or oid not in retained:
                        hist_exclusions.append(dict(source_panel="tierney_historical_calibration",
                                               target_cycle=target_cycle, cycle=int(r.cycle),
                                               horizon_days=h, extra_lag_days_beyond_native_proxy=lag,
                                               operator="tierney_historical_latest_per_14day_bin",
                                               observation_id=oid, sample_id=str(r.sample_id), poll_id="", question_id="",
                                               race_id=str(r.race_id),
                                               exclusion_reason=("superseded_within_fixed_14day_bin" if in_window
                                                                 else "outside_horizon_max_age_or_extra_lag")))

    # Exact target-race list intersection, without outcome or forecast values.
    support_rows = []
    for cycle in [2018, 2020, 2022, 2024]:
        roster_ids = set(roster.loc[roster.cycle.eq(cycle), "race_id"].astype(str))
        bank = bank_by_cycle[cycle]
        for rid in sorted(roster_ids | bank):
            support_rows.append(dict(cycle=cycle, race_id=rid, exact_v2_poll_roster=rid in roster_ids,
                                     exact_f0_bank_target=rid in bank, in_intersection=rid in roster_ids and rid in bank,
                                     status="common" if rid in roster_ids and rid in bank else
                                     "bank_only_no_exact_v2_poll_roster" if rid in bank else "roster_only"))
    hist_races = set(hist.race_id.astype(str))
    f0_training_rows = []
    for target_cycle, b in sorted(f0_banks.items()):
        for rid in sorted(set(b["training_race_ids"])):
            f0_training_rows.append(dict(target_cycle=target_cycle, race_id=rid,
                                         in_tierney_historical_panel=rid in hist_races,
                                         cycle_in_tierney_source=[c for c in [2012, 2016, 2018, 2020]
                                                                  if c < target_cycle and rid.startswith(f"{c}:")],
                                         status="poll_rows_exist_in_historical_panel" if rid in hist_races
                                         else "no_exact_poll_row_in_tierney_historical_panel"))
    pd.DataFrame(cells + hist_cells).to_csv(OUT / "coverage-cells.csv", index=False)
    pd.DataFrame(selected_rows + hist_selection_rows).to_csv(OUT / "operator-selection-rows.csv", index=False)
    pd.DataFrame(exclusions + hist_exclusions).to_csv(OUT / "selection-exclusions.csv", index=False)
    pd.DataFrame(helper_audits).to_csv(OUT / "538-helper-question-audits.csv", index=False)
    pd.DataFrame(support_rows).to_csv(OUT / "f0-race-support.csv", index=False)
    pd.DataFrame(f0_training_rows).to_csv(OUT / "f0-training-race-support.csv", index=False)
    pd.DataFrame(race_horizon).to_csv(OUT / "race-horizon-support.csv", index=False)

    pins = [SELECTION, F0_PLAN, BANK_COMPAT, V2 / "artifact-manifest.json", V2 / "plan.json", V2 / "receipt.json",
            V2 / "identity.py", V2 / "reviewed-crosswalk.csv", LAB / "momentum-identity-intake-luna-v2/receipt.json",
            ARCHIVE, ROSTER_A, ROSTER_B, HIST, BASE / "historical-intake-sol/README.md",
            INTEG / "src/forecasting/senate_data.py", INTEG / "src/forecasting/poll_data.py",
            BASE / "forecast-v2/source-snapshots/01-standard-plan-root-v1.json", BASE / "standard_runner.py"]
    receipt = dict(
        schema="MomentumPreflightCoverageLunaV1", root_selection_sha256=sha(SELECTION),
        expected_root_selection_sha256="5de976e5facbc84aa9db48e64fcc43864d1b8741fc5d1ddc134cc2e0fa05e5e9",
        f0_target_support_plan_sha256=sha(F0_PLAN),
        source_pins=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in pins],
        target_cycles=[2018, 2020, 2022, 2024], historical_source_cycles=[2012, 2016, 2018, 2020],
        horizons_days=list(HORIZONS), extra_lags_days_beyond_native_proxy=list(LAGS), max_age_days=MAX_AGE,
        cutoff_rule="Election Day minus H days at 00:00 UTC; require field end and native availability proxy plus extra lag <= cutoff before question selection.",
        availability_proxies=dict(
            target_538="aggregator created_at normalized to date plus one day; unverified original publication/version timestamp",
            historical_tierney="field-end date only; no original publication timestamp"),
        strict_asof_qualified=False, fit=False, score=False, optimization=False, new_poll_acquisition=False,
        source_mutation_outside_output=False,
        target_operators=["538_native_full_series", "538_latest_per_race_pollster"],
        calibration_operators=["tierney_native_full_series", "tierney_historical_latest_per_14day_bin"],
        question_operator="Exact senate_polls_as_of / normalize_538 helper; filter before question selection; exact nominee candidate-ID gate; one question per sample using LV > RV > adults, largest lineup, numeric question ID tie break.",
        latest_operator="Sort end date, availability proxy, sample ID, observation ID; keep last per race and pollster. Missing pollster rows are not merged and use sample-specific keys.",
        historical_bin_operator="Lead=floor(days_to_election/14); sort end_date, observation_id; keep last per cycle/race/pollster/bin exactly as standard_runner.py.",
        outputs=["coverage-cells.csv", "operator-selection-rows.csv", "selection-exclusions.csv",
                 "538-helper-question-audits.csv", "f0-race-support.csv",
                 "f0-training-race-support.csv", "race-horizon-support.csv"])
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    write_report(cells, hist_cells, support_rows)


def write_report(cells, hist_cells, support):
    df = pd.DataFrame(cells)
    rows = [
        "# Momentum coverage preflight (no fit)", "",
        "This packet follows the root-selected no-fit scope. It profiles the 2018–2024 538 target archive and the older Tierney/Volfovsky calibration panel separately. It does not fit, score, optimize, acquire polls, mutate source artifacts, or choose the eventual forecast operator.", "",
        "The 538 native helper is reapplied after each horizon and availability filter, before its question-selection rules. A separate latest-per-race/pollster operator is profiled. The historical calibration operator retains the latest row per race, pollster and fixed 14-day lead bin. The sources are never concatenated.", "",
        "## Cutoff and availability rules", "",
        "H30/H14/H7 are midnight UTC on the date 30/14/7 days before Election Day. Each row must end by that cutoff and be no more than 100 days from Election Day. The additional 0/1/3/7-day lag is added to each source's native availability proxy before filtering. For 538 this is aggregator created_at, normalized to date, plus one day; for the Tierney panel it is field end. Neither establishes that the exact historical poll was publicly available then. All outputs remain strict_asof_qualified=false.", "",
        "The exact question helper is applied after filtering. Candidate-answer rows are collapsed into a sample/question poll observation. The ID-level selection and exclusion file preserves every historical-panel row and each helper-admitted 538 observation used in the horizon operators; helper reason counts are retained separately. It does not claim a complete raw-candidate-answer exclusion ledger for questions rejected before helper output. The reviewed v2 2022 identity crosswalk is pinned unchanged; this coverage pass does not use election outcomes or add aliases.", "",
        "## Reading the coverage cells", "",
        "Each cell reports rows, unique samples, races, named pollsters, repeated race/pollster groups with at least two distinct field-end dates, their time spans, and interval metadata. Reported interval overlap only means the date ranges overlap; it does not reveal respondent overlap. Unknown pollster labels are not grouped together. Row counts are not independent poll counts when a source sample spans multiple races.", "",
        "The latest-per-pollster rule removes all within-pollster repeated observations at a given cutoff by construction. The fixed-bin calibration operator retains at most one poll per pollster per race per 14-day lead bin. These are distinct selection operators; a time-trend contrast under one cannot be assumed to exist under the other.", "",
        "## Support", "",
    ]
    sf = pd.DataFrame(support)
    for c in [2018, 2020, 2022, 2024]:
        common = sf[(sf.cycle == c) & sf.in_intersection]
        bank_n = int(sf[(sf.cycle == c) & sf.exact_f0_bank_target].race_id.nunique())
        roster_n = int(sf[(sf.cycle == c) & sf.exact_v2_poll_roster].race_id.nunique())
        rows.append(f"- {c}: eligible v2 poll roster {roster_n}; F0 target list {bank_n}; exact shared race IDs {len(common)}.")
    rows += ["", "The F0 list is a candidate support comparison only. It does not establish that the F0 model law or its upstream source rows are compatible with a polling update; Sol's separate read-only bank review covers that question. Detailed common and unmatched target race IDs are in f0-race-support.csv; fold-specific F0 training-race intersections with the Tierney panel are in f0-training-race-support.csv.", "",
        "The reviewed F0 target design and source features are cutoff-bound at H14. H7 can retain those older H14 features as such. An H30 poll-coverage cell is not a fully matched H30 F0 feature forecast; these tables report poll-source and race coverage only.", "",
        "## Known data limits", "",
        "The Tierney panel has no field starts, exact publication timestamps, candidate names/IDs, verified sample IDs, or original release chronology. The 538 archive contains reported start/end dates and poll/sample/question IDs but is a later historical archive without a verified publication/revision ledger. Long rolling periods, including the 2022 ActiVote record with a 91-day reported interval, do not identify when respondents were interviewed. Do not expand a long interval into daily polls or infer respondent dates from its midpoint. race-horizon-support.csv makes each no-sample bank/roster race visible at every cutoff.", "",
        "See receipt.json for source hashes and byte lengths, including the root selection, v2 reviewed identity packet, exact source helpers, target archive, rosters, historical panel and standard fixed-bin runner. No forecast result or model qualification is included.", "",
        "## H14, zero additional lag: target operator snapshot", ""]
    if not df.empty:
        for c in [2018, 2020, 2022, 2024]:
            rows.append(f"### {c}")
            for op in ["538_native_full_series", "538_latest_per_race_pollster"]:
                r = df[(df.cycle == c) & (df.horizon_days == 14) &
                       (df.extra_lag_days_beyond_native_proxy == 0) & (df.operator == op)]
                if len(r):
                    x = r.iloc[0]
                    rows.append(f"- {op}: {int(x.rows)} observations, {int(x.races)} races, {int(x.pollsters_known)} named pollsters, {int(x.repeated_race_pollster_groups)} repeated pollster/race groups; {int(x.bank_races_supported)}/{int(x.bank_races)} F0 target races represented.")
            rows.append("")
    rows.append("All H30/H14/H7 and extra-lag cells, plus complete selected/excluded IDs, are in the CSV artifacts. Root retains selection of a forecast target/operator.")
    (OUT / "REPORT.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
