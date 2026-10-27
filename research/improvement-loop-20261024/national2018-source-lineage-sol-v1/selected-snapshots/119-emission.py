"""Versioned report-label contamination channel; never creates ballot candidates."""
from dataclasses import dataclass
import json
import numpy as np
from src.senate_count_stack_v1.measurement import answer_law, answer_tensor, masked_tensor

VERSION='candidate_report_label_contamination_v1'
STATUS=('did_not_vote_this_contest','not_sure')

def ids(x):
 if type(x) is not list or not x or any(type(v) is not str or not v or v!=v.strip() for v in x) or len(set(x))!=len(x):raise ValueError('Unique nonempty literal axis required')
 return tuple(x)
def probability(x):
 if type(x) not in (int,float) or not np.isfinite(x) or not 0<=x<=1:raise ValueError('Finite probability scalar required')
 return float(x)
def immutable(a):
 a=np.ascontiguousarray(a,dtype=float);return np.frombuffer(a.tobytes(),dtype=float).reshape(a.shape)
@dataclass(frozen=True)
class Channel:
 canonical_json: str
 @property
 def config(self):return json.loads(self.canonical_json)
 @property
 def candidates(self):return tuple(self.config['candidate_ids'])
 @property
 def labels(self):return tuple(self.config['report_label_ids'])
 @property
 def mapping(self):
  c=self.config;return immutable([[float(label==target) for label in c['report_label_ids']] for target in c['candidate_default_labels']])
 @property
 def contamination(self):return immutable(self.config['contamination_probabilities'])

def validate_channel(config):
 if type(config) is Channel:config=config.config
 fields={'schema','source_id','candidate_ids','report_label_ids','candidate_default_labels','contamination_probabilities','epsilon'}
 if type(config) is not dict or set(config)!=fields or config['schema']!=VERSION:raise ValueError('Exact versioned channel fields required')
 ids([config['source_id']]);candidates=ids(config['candidate_ids']);labels=ids(config['report_label_ids'])
 if set(labels)&set(STATUS):raise ValueError('Candidate report labels cannot replace no-vote/not-sure states')
 m=config['candidate_default_labels']
 if type(m) is not list or len(m)!=len(candidates) or any(x not in labels for x in m):raise ValueError('Every actual candidate needs one explicit default report label')
 rho=config['contamination_probabilities']
 if type(rho) is not list or len(rho)!=len(labels):raise ValueError('Exact contamination label axis required')
 rho=[probability(x) for x in rho]
 if not np.isclose(sum(rho),1,atol=1e-12,rtol=0):raise ValueError('Normalized contamination simplex required')
 # Full label support prevents an emission-only label remaining impossible under beta mode.
 if any(x<=0 for x in rho):raise ValueError('Contamination must support every declared report label')
 e=config['epsilon']
 if type(e) is not dict:raise ValueError('Explicit epsilon configuration required')
 if e.get('kind')=='fixed':
  if set(e)!={'kind','value'}:raise ValueError('Exact fixed-epsilon config required')
  probability(e['value'])
 elif e.get('kind')=='beta':
  if set(e)!={'kind','alpha','beta','sharing_key'}:raise ValueError('Exact proper Beta source-sharing config required')
  ids([e['sharing_key']])
  for v in [e['alpha'],e['beta']]:
   if type(v) not in (int,float) or not np.isfinite(v) or v<=0:raise ValueError('Positive finite Beta shapes required')
 else:raise ValueError('Supported epsilon modes: fixed or beta')
 return Channel(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False))

def report_probabilities_tensor(candidate_probabilities,mapping,contamination,epsilon,*,array_module=np):
 """p[...,K] -> q[...,G]. Inputs validated upstream; scalar epsilon."""
 xp=array_module
 intended=xp.dot(candidate_probabilities,mapping)
 return (1-epsilon)*intended+epsilon*contamination

def emission_tensor(community,reported_turnout,candidate_probabilities,response,not_sure,false_vote_report,channel,epsilon,*,array_module=np):
 channel=validate_channel(channel)
 q=report_probabilities_tensor(candidate_probabilities,channel.mapping,channel.contamination,epsilon,array_module=array_module)
 return answer_tensor(community,reported_turnout,q,response,not_sure,false_vote_report,array_module=array_module)

def emission_law(*,demographic_ids,community_ids,community,reported_turnout,candidate_probabilities,response,not_sure,false_vote_report,channel,epsilon=None):
 channel=validate_channel(channel);config=channel.config
 if epsilon is None:
  if config['epsilon']['kind']!='fixed':raise ValueError('Supply shared epsilon draw for Beta channel; prior mean is not inference')
  epsilon=config['epsilon']['value']
 epsilon=probability(epsilon)
 if config['epsilon']['kind']=='fixed' and epsilon!=config['epsilon']['value']:raise ValueError('Fixed epsilon cannot be overridden')
 p=np.asarray(candidate_probabilities,dtype=float)
 if p.shape!=(len(demographic_ids),len(community_ids),len(channel.candidates)) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(-1),1,atol=1e-12,rtol=0):raise ValueError('Actual candidate probabilities must normalize on exact named axes')
 q=report_probabilities_tensor(p,channel.mapping,channel.contamination,epsilon)
 return answer_law(demographic_ids=demographic_ids,community_ids=community_ids,candidate_group_ids=channel.labels,community=community,reported_turnout=reported_turnout,candidate_groups=q,response=response,not_sure=not_sure,false_vote_report=false_vote_report)

def validate_observation_rows(rows,*,demographic_ids,community_ids,channel):
 channel=validate_channel(channel);seen=set()
 if type(rows) is not list or not rows:raise ValueError('Nonempty original respondent list required')
 for row in rows:
  if type(row) is not dict or set(row)!={'record_id','cell_id','community_mask','turnout_mask','answer_mask'}:raise ValueError('Exact one-person observed-pattern fields required')
  rid=row['record_id']
  if type(rid) is not str or not rid.strip() or rid in seen:raise ValueError('Unique original respondent ownership required')
  seen.add(rid)
  if row['cell_id'] not in demographic_ids:raise ValueError('Unknown demographic cell')
  for name,n in [('community_mask',len(community_ids)),('turnout_mask',2),('answer_mask',2+len(channel.labels))]:
   mask=row[name]
   if type(mask) is not list or len(mask)!=n or any(type(x) is not bool for x in mask) or not any(mask):raise ValueError('Nonempty Boolean mask on exact source axis required')
 return tuple(seen)

def shared_epsilon(channel,cache,*,pm_module):
 """Allocate one prior per declared source sharing key inside a model build.

 Pass one fresh cache per model, shared across every date/contest in this source.
 The cache prevents duplicate epsilon priors and rejects contradictory configs.
 """
 channel=validate_channel(channel);e=channel.config['epsilon']
 if e['kind']=='fixed':return e['value']
 key=e['sharing_key'];signature=(channel.config['source_id'],e['alpha'],e['beta'])
 if key in cache:
  if cache[key][0]!=signature:raise ValueError('Conflicting shared-source epsilon prior')
  return cache[key][1]
 value=pm_module.Beta('source_answer_error_'+key,alpha=e['alpha'],beta=e['beta'])
 cache[key]=(signature,value);return value

def survey_factor(scope,*,factor_name,rows,demographic_ids,community_ids,channel,community,reported_turnout,candidate_probabilities,response,not_sure,false_vote_report,epsilon,pm_module,array_module):
 """Leaf adapter to modular Scope.factor. Whole-Problem validator v2 is external.

 Caller owns source nuisance generation and the rest of measurement role. This
 replaces the corresponding survey factor; it must not be added beside it.
 """
 channel=validate_channel(channel)
 validate_observation_rows(rows,demographic_ids=demographic_ids,community_ids=community_ids,channel=channel)
 tokens=[f"survey:{channel.config['source_id']}:{r['record_id']}" for r in rows]
 # Detect local duplicate tokens too; whole-compiler ledger detects other stages.
 for claim in scope.factors.values():
  if set(tokens)&set(claim['tokens']):raise ValueError('Respondent already consumed by this stage')
 table=emission_tensor(community,reported_turnout,candidate_probabilities,response,not_sure,false_vote_report,channel,epsilon,array_module=array_module)
 indices=np.array([demographic_ids.index(r['cell_id']) for r in rows])
 probability=masked_tensor(table,indices,np.array([r['community_mask'] for r in rows]),np.array([r['turnout_mask'] for r in rows]),np.array([r['answer_mask'] for r in rows]),array_module=array_module)
 scope.factor(factor_name,tokens)
 return pm_module.Potential(factor_name,array_module.sum(array_module.log(probability)))
