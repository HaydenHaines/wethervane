"""Saved summary CSVs only; no predictions/banks/model imports."""
import csv,hashlib,json,math
from fractions import Fraction as F
from collections import defaultdict
from pathlib import Path
ROOT=Path('/home/hayden/projects/ballotmancy-house-benchmark');study=ROOT/'research/house-historical-interval-direction-20261026-v1';bench=ROOT/'research/house-polling-benchmark-20261024'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((study/'analysis-v1.json').read_text());pins={}
for item in a['inputs']:
 p=ROOT/item['path'];assert sha(p)==item['sha256'];pins[str(p)]=sha(p)
receipt=json.loads((bench/'results-v2/receipt.json').read_text());review=json.loads((bench/'output-audit-v2/review.json').read_text());assert review['status']=='passed';assert review['results']==a['inputs'][0];assert receipt['no_independent_holdout_claim']is True
for item in [receipt['protocol'],receipt['source'],review['source']]:
 p=ROOT/item['path'];assert sha(p)==item['sha256'];pins[str(p)]=sha(p)
for name in ['README.md','protocol-v1.json','admission-v1.json','label-access-clarification-v1.json']:
 p=bench/name;pins[str(p)]=sha(p)
for name in ['README.md','analysis-v1.json','test_analyze.py']:p=study/name;pins[str(p)]=sha(p)
with (bench/'results-v2/share-race-scores.csv').open()as stream:allrows=list(csv.DictReader(stream))
selected=defaultdict(list)
keys={(str(c['cycle']),c['implementation'],c['variant'])for c in a['cells']};recalculated={};max_difference=0
for r in allrows:
 key=(r['cycle'],r['implementation'],r['variant'])
 if key in keys and r['polled_primary']=='True':
  assert r['race_numerical_passed']=='True';selected[key].append(r)
for cell in a['cells']:
 key=(str(cell['cycle']),cell['implementation'],cell['variant']);rows=selected[key];assert len({r['race_id']for r in rows})==len(rows)>0;n=len(rows)
 below=sum(F(r['actual_dem_share'])<F(r['lower80'])for r in rows);above=sum(F(r['actual_dem_share'])>F(r['upper80'])for r in rows)
 stats={'races':n,'actual_below_interval':below,'actual_above_interval':above,'actual_inside_interval':n-below-above,'coverage80':float(F(n-below-above,n)),'mean_prediction_minus_actual':float(sum((F(r['predicted_dem_share'])-F(r['actual_dem_share'])for r in rows),F(0))/n),'mean_interval_width80':float(sum((F(r['upper80'])-F(r['lower80'])for r in rows),F(0))/n)}
 for field,v in stats.items():
  difference=abs(v-cell[field]);max_difference=max(max_difference,difference);assert difference<1e-12
 recalculated[key]=stats
with (bench/'results-v2/share-cycle-scores.csv').open()as stream:cycle_rows=list(csv.DictReader(stream))
for key,stats in recalculated.items():
 matches=[x for x in cycle_rows if (x['cycle'],x['implementation'],x['variant'])==key and x['subset']=='same_polled_races_F_P_FP'];assert len(matches)==1
 x=matches[0];assert int(x['n'])==stats['races'];assert abs(float(x['coverage80'])-stats['coverage80'])<1e-12;assert abs(float(x['width80'])-stats['mean_interval_width80'])<1e-12
cohorts={cycle:{tuple(sorted((r['race_id'],r['actual_dem_share'])for r in selected[k]))for k in selected if k[0]==cycle}for cycle in ['2018','2020','2022','2024']}
assert all(len(x)==1 for x in cohorts.values())
p_identical={}
for cycle in cohorts:
 sigs={tuple(sorted(tuple(r[k]for k in ['race_id','predicted_dem_share','lower80','upper80'])for r in selected[key]))for key in selected if key[0]==cycle and key[2]=='P'};p_identical[cycle]=len(sigs)==1
history=[{'cycle':cycle,'variant':variant,**recalculated[(str(cycle),'senate_history_refreshed_v1',variant)]}for cycle in [2018,2020,2022,2024]for variant in ['F','P','primary']]
out={'status':'saved_summary_calculations_reproduced_not_qualified','cells':len(a['cells']),'selected_summary_rows':sum(map(len,selected.values())),'max_fraction_vs_saved_float_difference':max_difference,'matched_outcome_sets_within_and_across_implementations':True,'polls_only_predictions_identical_across_implementations_by_cycle':p_identical,'history_refreshed_cells':history,'raw_predictive_rows_read':0,'pins':pins}
print(json.dumps(out,indent=2))
