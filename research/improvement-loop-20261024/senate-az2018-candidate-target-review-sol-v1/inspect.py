"""Small JSON/source inspection only; never reads banks or predictive rows."""
from pathlib import Path
import json,hashlib
b=Path('/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020')
owner=b/'national-2018-gpu-preparation-root-v1'
paths=[owner/'model-calibration-investigation-handoff-v1.md',owner/'arizona-writein-score-review-v1.json',owner/'candidate-calibration-review-v1/README.md',owner/'candidate-calibration-review-v1/diagnostics-v1.json',owner/'gpu-production-v1/attempt1/source-snapshots/117-components.py',b/'national-multicycle-execution-astra-v1/A2018/context.json',b/'historical-count-labels-review-root-v1/bound-v1/labels-2018-literal_first_rank.json',b/'historical-national-count-labels-sol-v2/source-packet-2018.json',b/'mixed-count-graph-astra-v1/prediction.py']
read=lambda p:json.loads(p.read_text())
c=read(paths[5]);l=read(paths[6]);s=read(paths[7]);r=read(paths[1]);cid='contest_az2018_regular';event='event_az2018_general';contest=next(x for x in c['problem']['contests']if x['id']==cid)
assert l['context_identity']==c['identity']==r['context_identity']
counts=l['contests'][cid]['observed']['candidate_counts'];assert counts==s['contests'][cid]['candidate_counts_by_estimand']['candidate_marks']
assert set(counts)=={x['id']for x in contest['candidates']}
joins={x['model_candidate_id']:x for x in s['contests'][cid]['joins']}
for x in r['writeins']:
 assert counts[x['candidate_id']]==x['observed_candidate_marks']==joins[x['candidate_id']]['source_record']['general_votes']
 assert joins[x['candidate_id']]['source_record']['party_literal']=='W'
 assert next(y for y in contest['candidates']if y['id']==x['candidate_id'])['bloc']=='other'
assert sum(x['observed_candidate_marks']for x in r['writeins'])==566
rows=[x for block in c['problem']['polls']for x in block['rows']if x['contest_id']==cid]
assert len(rows)==20
assert not [x for x in c['problem']['surveys']if x['contest_id']==cid]
assert not c['problem']['official_counts'] and not c['problem']['participation_reports']
inv=c['descriptor']['source_inventory'];prediction=b/'mixed-count-graph-astra-v1/prediction.py'
result={'context_identity':c['identity'],'candidate_count':len(contest['candidates']),'model_other_count':sum(x['bloc']=='other'for x in contest['candidates']),'candidate_ids':list(counts),'writein_observed_sum':566,'observed_named_sum':sum(counts.values()),'poll_rows':len(rows),'target_surveys':0,'context_official_count_factors':0,'context_participation_reports':0,'open_person':c['extension']['open_person'],'rule':contest['rule'],'current_prediction_matches_context_source_inventory':hashlib.sha256(prediction.read_bytes()).hexdigest()==inv[str(prediction.relative_to(b.parent.parent))]if str(prediction.relative_to(b.parent.parent))in inv else None,'pins':{str(p):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}for p in paths}}
print(json.dumps(result,indent=2))
