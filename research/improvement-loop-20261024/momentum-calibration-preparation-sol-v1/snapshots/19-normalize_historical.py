"""Build allowed-cycle Senate development inputs without fitting or scoring.

Run with the repository .venv/bin/python. Poll exports contain no election
outcomes. The separate outcomes export is restricted to 2012/2016/2018/2020.
CNN D/R integers are preserved as author-derived values, never raw counts.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.forecasting.senate_data import senate_polls_as_of

CYCLES = (2012, 2016, 2018, 2020)
RUNTIME = ROOT / '.redesign-runtime/response-20261007/senate'
RAW = HERE / 'raw'
OUT = HERE / 'derived'


def digest(path):
    data = path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                git_blob_sha1=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest())


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False)+'\n')


def allowed_parquet(name):
    path = RUNTIME / name
    columns = pq.ParquetFile(path).schema.names
    year = 'cycle' if 'cycle' in columns else 'year'
    frame = pd.read_parquet(path, filters=[(year, 'in', list(CYCLES))])
    assert set(frame[year]).issubset(CYCLES)
    return frame


def normalize_author(path, source, roster):
    # Author outcome columns are excluded at ingestion, including for older cycles.
    columns = ['election','state','year','pollName','startDate','endDate',
               'numberOfRespondents','democratic','republican','populationType','Mode','electionDate']
    frame = pd.read_csv(path, sep='\t' if path.suffix == '.tsv' else ',', usecols=columns)
    frame['source_row_number'] = np.arange(len(frame)) + 2
    coverage = frame.loc[frame.election.eq('Sen')].groupby('year').size().to_dict()
    frame = frame[frame.election.eq('Sen') & frame.year.isin(CYCLES)].copy()
    eligible = roster[roster.eligible].copy()
    assert not eligible.duplicated(['cycle','state']).any()
    joined = frame.merge(eligible[['cycle','state','race_id','source_race_id','dem_candidate_id',
                                 'rep_candidate_id','dem_candidate_name','rep_candidate_name','election_date']],
                         left_on=['year','state'], right_on=['cycle','state'], how='left', validate='many_to_one')
    start = pd.to_datetime(joined.startDate, errors='coerce')
    end = pd.to_datetime(joined.endDate, errors='coerce')
    source_election = pd.to_datetime(joined.electionDate, errors='coerce')
    election = source_election.copy()
    roster_date = pd.to_datetime(joined.election_date, errors='coerce')
    # The pinned author file assigns every 2016 row November 3. The admitted
    # roster records the November 8 general election. Correct only this exact
    # documented metadata error, retaining its source date and correction flag.
    correction = (joined.year.eq(2016) & source_election.eq(pd.Timestamp('2016-11-03'))
                  & roster_date.eq(pd.Timestamp('2016-11-08')))
    election.loc[correction] = roster_date[correction]
    d = pd.to_numeric(joined.democratic, errors='coerce')
    r = pd.to_numeric(joined.republican, errors='coerce')
    n = pd.to_numeric(joined.numberOfRespondents, errors='coerce')
    reason = pd.Series('', index=joined.index)
    reason.loc[joined.race_id.isna()] = 'outside_eligible_roster'
    reason.loc[reason.eq('') & (end.isna() | election.isna() | end.ge(election))] = 'invalid_field_end_or_election'
    reason.loc[reason.eq('') & election.ne(roster_date)] = 'election_date_not_roster_date'
    reason.loc[reason.eq('') & (d.le(0) | r.le(0) | (d+r).isna() | n.le(0) | n.isna())] = 'invalid_source_values'
    if source == 'stanford_auxiliary':
        reason.loc[reason.eq('') & (d+r).gt(100.01)] = 'invalid_percentage_sum'
    audit = dict(source=source, source_senate_cycle_rows=coverage,
                 allowed_rows=len(frame), exclusions=dict(Counter(reason[reason.ne('')])) )
    keep = reason.eq('')
    x = joined[keep].copy()
    x['cycle'] = x.cycle.astype(int)
    x['start_date'] = start[keep]
    x['end_date'] = end[keep]
    x['election_date'] = election[keep]
    x['source_election_date'] = source_election[keep]
    x['election_date_correction'] = np.where(correction[keep],
         'source_2016_Nov03_corrected_to_admitted_roster_Nov08', 'none')
    x['source_democratic_value'] = d[keep]
    x['source_republican_value'] = r[keep]
    x['reported_n'] = n[keep]
    x['dem_share'] = d[keep] / (d[keep]+r[keep])
    x['first_candidate_conditional_share'] = x.dem_share
    x['first_candidate_id'] = x.dem_candidate_id
    x['second_candidate_id'] = x.rep_candidate_id
    x['first_candidate_name'] = x.dem_candidate_name
    x['second_candidate_name'] = x.rep_candidate_name
    x['source_id'] = source
    x['source_path'] = str(path.relative_to(ROOT))
    x['source_value_basis'] = ('author_derived_D_R_values_not_verified_raw_counts' if source == 'tierney_cnn'
                               else 'released_percentage_transcription')
    x['reported_n_basis'] = ('author_two_party_total_original_nominal_sample_unknown' if source == 'tierney_cnn'
                             else 'reported_respondents_design_effect_unknown')
    x['effective_n'] = np.nan
    x['variance'] = np.nan
    x['pollster'] = x.pollName
    x['population'] = x.populationType.fillna('unknown')
    x['methodology'] = x.Mode.fillna('unknown')
    x['publication_date'] = pd.NaT
    x['available_date'] = x.end_date
    x['availability_basis'] = 'field_end_date_proxy_only_NOT_verified_publication'
    x['publication_date_verified'] = False
    x['historical_version_verified'] = False
    x['strict_asof_qualified'] = False
    x['candidate_join_basis'] = 'unique_eligible_state_cycle_D_R_roster_pair_no_author_candidate_names'
    x['source_candidate_names_verified'] = False
    x['sample_overlap_status'] = 'unknown_no_author_sample_ID_or_field_start'
    # Exact copied rows are collapsed; conflicting same-date estimates remain
    # explicit unknown-overlap rows, not asserted independent samples.
    semantic = ['cycle','state','pollster','end_date','source_democratic_value','source_republican_value','reported_n']
    before = len(x)
    x = x.drop_duplicates(semantic).copy()
    x['observation_id'] = source+':row:'+x.source_row_number.astype(str)
    x['sample_id'] = 'unknown:'+x.observation_id
    x['potential_overlap_group'] = source+':'+x.cycle.astype(str)+':'+x.pollster.astype(str)
    x['days_to_election'] = (x.election_date-x.end_date).dt.days
    x['forecast_target'] = 'Democratic_share_conditional_on_D_R'
    audit.update(retained_rows=len(x), exact_duplicate_rows_removed=before-len(x),
                 retained_rows_by_cycle=x.groupby('cycle').size().to_dict(),
                 retained_races_by_cycle=x.groupby('cycle').race_id.nunique().to_dict(),
                 missing_start_dates=int(x.start_date.isna().sum()),
                 missing_population=int(x.population.eq('unknown').sum()),
                 missing_mode=int(x.methodology.eq('unknown').sum()),
                 duplicate_pollster_race_end_groups=int(x.groupby(['cycle','race_id','pollster','end_date']).size().gt(1).sum()),
                 author_D_plus_R_equals_reported_n_rows=int((x.source_democratic_value+x.source_republican_value).eq(x.reported_n).sum()))
    audit['source_election_date_corrections'] = int(correction[keep].sum())
    drop = ['election','year','pollName','startDate','endDate','numberOfRespondents',
            'democratic','republican','populationType','Mode','electionDate']
    return x.drop(columns=drop).sort_values(['cycle','race_id','end_date','observation_id']), audit


def main():
    OUT.mkdir(exist_ok=True)
    roster = allowed_parquet('race_roster.parquet')
    features = allowed_parquet('race_inputs.parquet')
    outcomes = allowed_parquet('historical_senate_returns.parquet')
    assert set(outcomes.cycle) == set(CYCLES)
    assert not outcomes.race_id.duplicated().any()
    assert not any('actual' in c or 'votes' in c for c in features.columns)
    features.to_csv(OUT/'race_inputs_allowed_cycles.csv', index=False)
    roster.to_csv(OUT/'race_roster_allowed_cycles.csv', index=False)
    outcomes.to_csv(OUT/'outcomes_allowed_cycles_SEPARATE.csv', index=False)
    author, author_audit = normalize_author(RAW/'tierney-polls_cnn_dataset.csv','tierney_cnn',roster)
    stanford, stanford_audit = normalize_author(RAW/'stanford-polls_auxiliary_dataset.tsv','stanford_auxiliary',roster)
    author.to_csv(OUT/'tierney_poll_catalog.csv', index=False)
    stanford.to_csv(OUT/'stanford_poll_catalog.csv', index=False)
    # Use metadata and allowed cycles only; never read retrospective quality ratings.
    archive = ROOT/'.redesign-runtime/forecast-20261007/polls/senate_polls_historical.csv'
    raw_columns = ['poll_id','question_id','cycle','state','party','pct','sample_size','population',
                   'start_date','end_date','created_at','race_id','stage','candidate_name','pollster',
                   'candidate_id','election_date','tracking','hypothetical','ranked_choice_reallocated',
                   'methodology','url','partisan']
    raw = pd.read_csv(archive, usecols=raw_columns, low_memory=False)
    archive_coverage = raw.groupby('cycle').size().to_dict()
    raw = raw[raw.cycle.isin([2018,2020])].copy()
    full = []
    cutoff_audits = []
    horizon_tables = []
    for cycle in [2018,2020]:
        election = pd.Timestamp({2018:'2018-11-06',2020:'2020-11-03'}[cycle])
        for horizon in [90,30,14,7,1]:
            cutoff = election-pd.Timedelta(days=horizon)
            polls, audit = senate_polls_as_of(raw,roster,cycle=cycle,cutoff=cutoff)
            audit.update(horizon=horizon)
            cutoff_audits.append(audit)
            if not polls.empty:
                polls['horizon'] = horizon
                polls['cutoff_date'] = cutoff
                polls['strict_asof_qualified'] = False
                polls['reported_n'] = polls.sample_size
                polls['reported_n_basis'] = 'reported_question_sample_design_effect_unknown'
                polls['first_candidate_conditional_share'] = polls.dem_share
                polls['source_value_basis'] = 'published_weighted_candidate_percentages'
                horizon_tables.append(polls)
                if horizon == 1:
                    full.append(polls.drop(columns=['horizon','cutoff_date']))
    cache_catalog = pd.concat(full,ignore_index=True)
    cache_catalog.to_csv(OUT/'cached_538_poll_catalog.csv',index=False)
    pd.concat(horizon_tables,ignore_index=True).to_csv(OUT/'cached_538_cutoff_specific_catalog.csv',index=False)
    # A single-source four-cycle panel prevents accidentally stacking author
    # copies and 538 releases as independent evidence.
    author.to_csv(OUT/'historical_poll_panel.csv', index=False)
    report = dict(schema='historical_senate_poll_intake_v1', cycles=list(CYCLES),
                  fitting=False, scoring=False, protected_outcomes_opened=False,
                  source_2022_2024_poll_toplines_not_exported=True,
                  author_audit=author_audit, stanford_audit=stanford_audit,
                  archive_candidate_row_coverage=archive_coverage,
                  cached_538_cutoff_audits=cutoff_audits,
                  eligible_races=roster[roster.eligible].groupby('cycle').size().to_dict(),
                  outcomes_by_cycle=outcomes.groupby('cycle').size().to_dict(),
                  author_race_joins_missing_outcome=int((~author.race_id.isin(outcomes.race_id)).sum()),
                  cutoff_coverage=[])
    for cycle,g in author.groupby('cycle'):
        election=pd.Timestamp(g.election_date.iloc[0])
        for horizon in [90,30,14,7]:
            selected=g[g.available_date.le(election-pd.Timedelta(days=horizon))]
            report['cutoff_coverage'].append(dict(cycle=int(cycle),horizon=horizon,
                    rows=len(selected),races=int(selected.race_id.nunique()),
                    basis='field_end_proxy_retrospective_development_only'))
    write_json(OUT/'intake_audit.json',report)
    pins=[digest(RAW/'tierney-polls_cnn_dataset.csv'),digest(RAW/'stanford-polls_auxiliary_dataset.tsv'),
          digest(archive),digest(RUNTIME/'race_roster.parquet'),digest(RUNTIME/'race_inputs.parquet'),
          digest(RUNTIME/'historical_senate_returns.parquet')]
    write_json(OUT/'input_hashes.json',pins)
    print(json.dumps(dict(author_rows=len(author),author_cycles=author_audit['retained_rows_by_cycle'],
                    author_races=author_audit['retained_races_by_cycle'],cached_538_rows=len(cache_catalog),
                    outcome_rows=len(outcomes),output=str(OUT.relative_to(ROOT))),indent=2))


if __name__ == '__main__':
    main()
