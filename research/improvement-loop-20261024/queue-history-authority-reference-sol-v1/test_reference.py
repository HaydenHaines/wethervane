import copy,itertools,json,os,shutil,sqlite3,subprocess,sys,tempfile,unittest
from pathlib import Path
from reference import Reference,Refused,canonical,check_response,TRUST

class ReferenceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name);self.r=Reference.create(self.base/'domain')
 def state(self):return self.r.status()
 def admit(self,name,deps=(),output=None,key=None):
  s=self.state();return self.r.admit(key or'admit-'+name,s['head'],s['lease'],name,output or str(self.base/'outputs'/name),deps,evidence=b'invented tiny bytes',target='invented-event',unit='invented-count')
 def reserve(self,name='a'):
  s=self.state();return self.r.reserve('reserve-'+name,s['head'],s['lease'],name,'invented-worker-'+name)
 def evidence(self,outcome='accepted_unqualified'):
  a=self.state()['active'];return {'schema':'InventedCleanupEvidenceV1','job':a['job'],'worker_token':a['worker_token'],'reservation_hash':a['reservation_hash'],'worker_status':'exited','cleanup_verified':True,'outcome':outcome,'terminal_sha256':'a'*64,'cleanup_sha256':'b'*64,'scientifically_qualified':False}
 def release(self,outcome='accepted_unqualified'):
  e=self.evidence(outcome);s=self.state();return self.r.release('release-'+e['job'],s['head'],s['lease'],e)
 def sql(self,query,args=()):
  with sqlite3.connect(self.r.path)as db:db.execute(query,args)
 def backup(self,path):
  with sqlite3.connect(self.r.path)as source,sqlite3.connect(path)as dest:source.backup(dest)
 def restore(self,path):
  with sqlite3.connect(path)as source,sqlite3.connect(self.r.path)as dest:source.backup(dest)
 def client(self,action,params,crash=None):
  code="import json,sys;from reference import Reference,Refused;r=Reference(sys.argv[1]);p=json.loads(sys.argv[3]);\ntry: print(json.dumps(getattr(r,sys.argv[2])(*p)))\nexcept Refused: sys.exit(2)"
  env=dict(os.environ)
  if crash:env['REFERENCE_CRASH']=crash
  return subprocess.run([sys.executable,'-B','-c',code,str(self.r.root),action,json.dumps(params)],env=env,capture_output=True,text=True,timeout=5)
 def test_empty_open_never_completed_and_nonempty_states(self):
  s=self.state();self.assertEqual(s['trust'],TRUST);self.assertTrue(s['intake_open']);self.assertFalse(s['nonempty_current_admissions_terminal']);self.assertFalse(s['closed_nonempty_batch_terminal'])
  with self.assertRaises(Refused):self.r.seal('empty',s['head'],s['lease'])
  self.admit('a');self.reserve();self.assertIsNotNone(self.state()['active']);self.release();s=self.state();self.assertTrue(s['nonempty_current_admissions_terminal']);self.assertTrue(s['all_current_admissions_accepted_unqualified']);self.assertFalse(s['closed_nonempty_batch_terminal']);self.assertFalse(s['runtime_worker_evidence_authenticated'])
  self.r.seal('close',s['head'],s['lease']);self.assertTrue(self.state()['closed_nonempty_batch_terminal']);self.assertFalse(self.state()['scientifically_qualified'])
  with self.assertRaises(Refused):self.admit('b')
 def test_exact_idempotency_and_changed_body_refusal(self):
  s=self.state();args=['key',s['head'],1,'a',str(self.base/'outputs/a')];first=self.r.admit(*args);self.assertEqual(self.r.admit(*args),first);self.assertEqual(self.state()['head']['sequence'],1)
  with self.assertRaisesRegex(Refused,'idempotency_body_conflict'):self.r.admit(*args,evidence=b'different')
  self.assertTrue(first['historical_commit_only']);self.assertFalse(first['live_launch_authorization']);self.assertFalse(first['runtime_worker_evidence_authenticated'])
 def test_ancestor_conflict_all_orders_and_root_alias(self):
  paths=[str(self.base/'out/a'),str(self.base/'out/a-b'),str(self.base/'out/a/b')]
  for n,order in enumerate(itertools.permutations(paths)):
   r=Reference.create(self.base/('order'+str(n)));rejections=0
   for i,path in enumerate(order):
    s=r.status()
    try:r.admit('key'+str(i),s['head'],s['lease'],'j'+str(i),path)
    except Refused:rejections+=1
   self.assertGreater(rejections,0)
  self.admit('a');s=self.state()
  for path in [str(self.base/'outputs/a'),str(self.base/'outputs/a/child'),str(self.base/'outputs'),'//alias','relative','/x/../y']:
   with self.assertRaises(Refused):self.r.admit('conflict',s['head'],1,'other',path)
 def test_missing_duplicate_dependencies_and_new_admission_after_success(self):
  with self.assertRaises(Refused):self.admit('a',['missing'])
  self.admit('a');self.reserve();self.release();self.admit('b',['a']);self.assertEqual(self.state()['ready'],['b']);self.assertFalse(self.state()['nonempty_current_admissions_terminal'])
  with self.assertRaises(Refused):self.admit('c',['a','a'])
 def test_unknown_worker_fake_cleanup_and_lease_never_release_capacity(self):
  self.admit('a');self.reserve();s=self.state();original=self.evidence()
  for field,value in [('worker_status','unknown'),('cleanup_verified',False),('job','foreign'),('worker_token','wrong'),('reservation_hash','0'*64),('scientifically_qualified',True),('terminal_sha256','bad')]:
   e=copy.deepcopy(original);e[field]=value
   with self.assertRaises(Refused):self.r.release('bad',s['head'],s['lease'],e)
   self.assertIsNotNone(self.state()['active'])
  self.r.advance_lease('new-lease',s['head'],1);self.assertIsNotNone(self.state()['active'])
  with self.assertRaisesRegex(Refused,'stale_head_or_lease'):self.r.release('old-controller',s['head'],1,original)
  fresh=self.state();self.r.release('new-controller',fresh['head'],fresh['lease'],original);self.assertIsNone(self.state()['active'])
 def test_multilevel_multibranch_failure_terminal_not_success(self):
  self.admit('a');self.admit('b',['a']);self.admit('c',['b']);self.admit('d',['a']);self.reserve();self.release('failed');s=self.state();self.assertEqual(s['states'],{'a':'failed','b':'blocked','c':'blocked','d':'blocked'});self.assertEqual(s['ready'],[]);self.assertTrue(s['nonempty_current_admissions_terminal']);self.assertFalse(s['all_current_admissions_accepted_unqualified']);self.assertFalse(s['closed_nonempty_batch_terminal'])
 def test_stale_new_request_and_capacity_one(self):
  initial=self.state();self.admit('a');self.admit('b');s=self.state();self.reserve()
  with self.assertRaises(Refused):self.r.reserve('second',s['head'],s['lease'],'b','worker-b')
  now=self.state()
  with self.assertRaisesRegex(Refused,'capacity_reserved'):self.r.reserve('second',now['head'],now['lease'],'b','worker-b')
  with self.assertRaises(Refused):self.r.advance_lease('stale',initial['head'],1)
 def test_crash_admission_boundaries_and_response_loss(self):
  for i,stage in enumerate(['after_event_insert','after_projection','before_commit','after_commit']):
   self.r=Reference.create(self.base/('crash'+str(i)));s=self.state();params=['key',s['head'],1,'a',str(self.base/'outputs/a')]
   child=self.client('admit',params,stage);self.assertEqual(child.returncode,93,child.stderr);state=self.state()
   self.assertEqual(len(state['states']),int(stage=='after_commit'))
   result=self.r.admit(*params);self.assertEqual(self.r.lookup('key'),result);self.assertEqual(self.state()['head']['sequence'],1)
   with sqlite3.connect(self.r.path)as db:self.assertEqual(db.execute('SELECT count(*) FROM objects').fetchone()[0],1)
 def test_release_crashes_before_after_commit_idempotent(self):
  for i,stage in enumerate(['before_commit','after_commit']):
   self.r=Reference.create(self.base/('releasecrash'+str(i)));self.admit('a');self.reserve();s=self.state();e=self.evidence();params=['release',s['head'],s['lease'],e]
   child=self.client('release',params,stage);self.assertEqual(child.returncode,93,child.stderr);self.assertEqual(self.state()['active']is None,stage=='after_commit')
   r=self.r.release(*params);self.assertEqual(self.r.lookup('release'),r);self.assertIsNone(self.state()['active'])
 def test_projection_corruption_and_rebuild_idempotency_across_crash(self):
  self.admit('a');s=self.state();self.sql('DELETE FROM ready')
  with self.assertRaisesRegex(Refused,'projection_corrupt'):self.state()
  self.assertIsNotNone(self.r.lookup('admit-a'))
  child=self.client('rebuild',[s['head'],s['lease']], 'before_rebuild_commit');self.assertEqual(child.returncode,93)
  with self.assertRaises(Refused):self.state()
  child=self.client('rebuild',[s['head'],s['lease']], 'after_rebuild_commit');self.assertEqual(child.returncode,93)
  self.assertEqual(self.state()['ready'],['a']);self.assertEqual(self.r.rebuild(s['head'],1),self.r.rebuild(s['head'],1));self.assertEqual(self.state()['head'],s['head'])
 def test_fake_accepted_projection_receipt_and_object_tamper_reject(self):
  self.admit('a');self.sql("UPDATE jobs SET body=?",(canonical({'state':'accepted_unqualified'}),))
  with self.assertRaisesRegex(Refused,'projection_corrupt'):self.state()
  h,lease=self.r.canonical_head();self.r.rebuild(h,lease);self.sql('UPDATE responses SET body=?',(canonical({'action':'release','historical_commit_only':True}),))
  with self.assertRaisesRegex(Refused,'idempotency_receipt_corrupt'):self.r.lookup('admit-a')
 def test_referenced_object_tamper_and_malformed_heads_reject(self):
  self.admit('a');self.sql('UPDATE objects SET body=?',(b'changed',))
  with self.assertRaisesRegex(Refused,'referenced_object_corrupt'):self.state()
  r=Reference.create(self.base/'heads');h=r.status()['head']
  for bad in [dict(h,sequence=True),dict(h,hash='bad'),dict(h,extra=True)]:
   with self.assertRaises(Refused):r.advance_lease('key',bad,1)
 def test_same_domain_same_sequence_fork_and_rollback_needs_caller_floor(self):
  saved=self.base/'genesis.sqlite';self.backup(saved);self.admit('a');a=self.state()['head'];self.restore(saved);self.admit('b');b=self.state()['head'];self.assertEqual(a['domain'],b['domain']);self.assertEqual(a['sequence'],b['sequence']);self.assertNotEqual(a['hash'],b['hash'])
  with self.assertRaisesRegex(Refused,'caller_floor_rollback_or_fork'):self.r.status(a)
  self.assertEqual(self.state()['rollback_floor'],'none_not_rollback_protected');self.restore(saved)
  with self.assertRaises(Refused):self.r.status(a)
  self.assertEqual(self.state()['head']['sequence'],0)
 def test_stale_nonce_response_and_unauthenticated_boundary(self):
  floor=self.state()['head'];reply=self.r.head_response('nonce-a',floor)
  with self.assertRaisesRegex(Refused,'nonce_or_domain'):check_response(reply,'nonce-b',floor['domain'],floor)
  r=check_response(reply,'nonce-a',floor['domain'],floor);self.assertFalse(r['authenticated']);self.assertFalse(reply['response_authenticated'])
 def test_source_version_domain_and_bounds(self):
  with sqlite3.connect(self.r.path)as db:
   config=json.loads(db.execute("SELECT body FROM meta WHERE k='config'").fetchone()[0]);bad=copy.deepcopy(config);bad['schema']='other';db.execute("UPDATE meta SET body=? WHERE k='config'",(canonical(bad),))
  with self.assertRaisesRegex(Refused,'domain_version'):self.state()
  self.sql("UPDATE meta SET body=? WHERE k='config'",(canonical(config),));bad=copy.deepcopy(config);bad['source']['sha256']='0'*64;self.sql("UPDATE meta SET body=? WHERE k='config'",(canonical(bad),))
  with self.assertRaisesRegex(Refused,'source_version'):self.state()
  self.sql("UPDATE meta SET body=? WHERE k='config'",(canonical(config),));s=self.state()
  with self.assertRaises(Refused):self.r.admit('big',s['head'],1,'a',str(self.base/'out/a'),evidence=b'x'*65537)
 def test_concurrent_reservation_one_commit_one_stale(self):
  self.admit('a');s=self.state();params=['reserve',s['head'],1,'a','invented-worker']
  code="import json,sys;from reference import Reference,Refused;r=Reference(sys.argv[1]);p=json.loads(sys.argv[2]);\ntry:r.reserve(*p)\nexcept Refused:sys.exit(2)"
  children=[]
  for n in range(2):
   p=list(params);p[0]='reserve'+str(n);children.append(subprocess.Popen([sys.executable,'-B','-c',code,str(self.r.root),json.dumps(p)],stdout=subprocess.PIPE,stderr=subprocess.PIPE))
  codes=[]
  for child in children:
   out,err=child.communicate(timeout=5);codes.append(child.returncode);self.assertIn(child.returncode,[0,2],err)
  self.assertEqual(sorted(codes),[0,2]);self.assertEqual(self.state()['states']['a'],'active')
if __name__=='__main__':unittest.main(verbosity=2)
