import copy,json,os,subprocess,sys,tempfile,unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import execution as ex
import experiment_queue as q
import test_experiment_queue as fixtures

class DecodeTests(unittest.TestCase):
 def test_exact_bytes_type_and_independent_mutable_objects(self):
  with ex.decode_operation()as memo:
   a=ex.decode(b'{"x":[1]}');a['x'].append(9);b=ex.decode(b'{"x":[1]}');self.assertEqual(b,{'x':[1]});b['x'].append(8);self.assertEqual(ex.decode(b'{"x":[1]}'),{'x':[1]})
   ex.decode('{"x":[1]}');self.assertEqual(memo.stats['parses'],2);self.assertEqual(memo.stats['hits'],2)
  self.assertTrue(memo.stats['cleared']);self.assertFalse(memo.values);self.assertIsNone(ex._DECODE_MEMO.get())
 def test_outside_operation_never_memoizes_and_new_scopes_start_empty(self):
  with patch.object(ex,'_base_decode',wraps=ex._base_decode)as parse:
   ex.decode(b'{}');ex.decode(b'{}');self.assertEqual(parse.call_count,2)
   for _ in range(2):
    with ex.decode_operation()as m:ex.decode(b'{}');ex.decode(b'{}');self.assertEqual(m.stats['hits'],1)
   self.assertEqual(parse.call_count,4)
 def test_success_only_invalid_duplicate_nonfinite_and_bad_encoding(self):
  bad=[b'{"x":1,"x":2}',b'NaN',b'Infinity',b'-Infinity',b'{',b'"\xff"']
  with ex.decode_operation()as m:
   for data in bad:
    for _ in range(2):
     try:ex._base_decode(data)
     except BaseException as base:
      with self.assertRaises(type(base)):ex.decode(data)
   self.assertFalse(m.values);self.assertEqual(m.stats['errors'],12)
   # Preserve existing JSON exponent-overflow behavior rather than change its law.
   self.assertEqual(ex.decode(b'1e400'),ex._base_decode(b'1e400'))
 def test_bypass_types_large_items_and_limits(self):
  with ex.decode_operation(max_entries=2,max_bytes=8,max_item_bytes=8)as m:
   ex.decode(bytearray(b'{}'));ex.decode(bytearray(b'{}'));self.assertEqual(m.stats['hits'],0)
   ex.decode(b'"123456789"');self.assertEqual(m.stats['bypasses'],3)
   for data in [b'{}',b'[]',b'null',b'false']:ex.decode(data)
   self.assertLessEqual(len(m.values),2);self.assertLessEqual(m.bytes,8);self.assertGreater(m.stats['evictions'],0)
  for kw in [{'max_entries':65},{'max_bytes':262145},{'max_item_bytes':65537},{'max_entries':True},{'max_bytes':-1}]:
   with self.assertRaises(ex.ExecutionError):
    with ex.decode_operation(**kw):pass
  with ex.decode_operation(max_entries=0)as m:ex.decode(b'{}');ex.decode(b'{}');self.assertEqual(m.stats['hits'],0)
 def test_unicode_surrogates_and_str_byte_budget(self):
  text='"'+chr(0xd800)+'"'
  with ex.decode_operation(max_item_bytes=4)as m:
   self.assertEqual(ex.decode(text),ex._base_decode(text));self.assertEqual(ex.decode(text),ex._base_decode(text));self.assertEqual(m.stats['hits'],0)
 def test_all_exit_cleanup_and_nested_scope_refusal(self):
  for exc in [ValueError,KeyboardInterrupt,SystemExit]:
   with self.assertRaises(exc):
    with ex.decode_operation()as m:
     ex.decode(b'{}');raise exc('invented')
   self.assertTrue(m.stats['cleared']);self.assertFalse(m.values);self.assertIsNone(ex._DECODE_MEMO.get())
  with ex.decode_operation():
   with self.assertRaises(ex.ExecutionError):
    with ex.decode_operation():pass
 def test_copy_failure_bypasses_without_new_parse_semantics(self):
  with ex.decode_operation()as m:
   with patch.object(ex,'deepcopy',side_effect=RecursionError):self.assertEqual(ex.decode(b'{"a":1}'),{'a':1})
   self.assertFalse(m.values);self.assertEqual(m.stats['copy_bypasses'],1)
   ex.decode(b'{"a":1}')
   with patch.object(ex,'deepcopy',side_effect=MemoryError):self.assertEqual(ex.decode(b'{"a":1}'),{'a':1})
   self.assertFalse(m.values)
 def test_deep_nesting_matches_original_or_bypasses(self):
  data=b'['*600+b'0'+b']'*600
  with ex.decode_operation()as m:
   expected=ex._base_decode(data);actual=ex.decode(data)
   # Iterative comparison avoids recursion in unittest's equality formatter.
   for _ in range(600):expected=expected[0];actual=actual[0]
   self.assertEqual(expected,actual);self.assertGreater(m.stats['copy_bypasses'],0)
 def test_fork_identity_cannot_reuse_parent_memo(self):
  with ex.decode_operation()as m:
   ex.decode(b'{}')
   with patch.object(ex.os,'getpid',return_value=m.owner_pid+1),patch.object(ex,'_base_decode',wraps=ex._base_decode)as base:
    ex.decode(b'{}');self.assertEqual(base.call_count,1)
   self.assertEqual(m.stats['hits'],0)
 def test_abrupt_exit_has_no_persisted_scope_on_restart(self):
  code="import execution as ex,os\nwith ex.decode_operation():\n ex.decode(b'{}')\n os._exit(93)\n"
  child=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,timeout=3);self.assertEqual(child.returncode,93)
  child=subprocess.run([sys.executable,'-B','-c',"import execution as ex; print(ex._DECODE_MEMO.get() is None)"],capture_output=True,timeout=3);self.assertEqual(child.returncode,0);self.assertEqual(child.stdout.strip(),b'True')

class OperationTests(unittest.TestCase):
 def setUp(self):
  self.fx=fixtures.QueueTests(methodName='runTest');self.fx.setUp();self.addCleanup(self.fx.doCleanups);self.root=self.fx.root
 def test_live_same_size_input_edit_and_namespace_checks_still_fresh(self):
  self.fx.admit([self.fx.job('tiny')]);self.fx.run_queue();_,fd=q.dispatcher(self.root)
  try:
   with ex.decode_operation()as m:
    first=ex.ownership_audit(self.root/'executions',q.ownership_expected(self.root));source=self.fx.source;raw=source.read_bytes();before=source.stat()
    try:
     source.write_bytes(raw.replace(b'import hashlib',b'import hashliB',1));os.utime(source,ns=(before.st_atime_ns,before.st_mtime_ns));fresh=ex.ownership_audit(self.root/'executions',q.ownership_expected(self.root));self.assertTrue(fresh['historical_input_drift']);self.assertEqual(first['evidence'],fresh['evidence'])
    finally:source.write_bytes(raw)
    path=next((self.root/'executions/jobs').glob('*/claim-*.json'));raw=path.read_bytes()
    try:
     path.write_bytes(b'{}')
     with self.assertRaises(Exception):ex.ownership_audit(self.root/'executions',q.ownership_expected(self.root))
    finally:path.write_bytes(raw)
    spec=self.fx.job('tiny')['execution_spec'];lock=ex.job_directory(self.root/'executions',spec['cycle_id'],spec['job_key'])/'lock';saved=self.fx.base/'saved-lock';lock.rename(saved)
    try:
     lock.write_bytes(saved.read_bytes())
     with self.assertRaisesRegex(ex.ExecutionError,'job lock identity changed'):ex.ownership_audit(self.root/'executions',q.ownership_expected(self.root))
    finally:os.replace(saved,lock)
  finally:os.close(fd)
 def test_cached_decode_does_not_skip_ownership_lock_contention(self):
  self.fx.admit([self.fx.job('lock')]);self.fx.run_queue();_,fd=q.dispatcher(self.root);holder=None
  try:
   with ex.decode_operation():
    ex.ownership_audit(self.root/'executions',q.ownership_expected(self.root))
    code="import fcntl,sys;f=open(sys.argv[1],'r+');fcntl.flock(f,fcntl.LOCK_EX);print('ready',flush=True);sys.stdin.read()"
    holder=subprocess.Popen([sys.executable,'-B','-c',code,str(self.root/'executions/ownership.lock')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    self.assertEqual(holder.stdout.readline().strip(),'ready')
    with self.assertRaises(ex.LockBusy):ex.ownership_audit(self.root/'executions',q.ownership_expected(self.root))
  finally:
   if holder is not None:
    holder.stdin.close();holder.wait(3);holder.stdout.close();holder.stderr.close()
   os.close(fd)
 def test_public_status_tick_finalize_scope_exits_and_new_scope(self):
  original=ex.decode_operation;seen=[]
  @contextmanager
  def tracked():
   with original()as m:
    seen.append(m);yield m
  with patch.object(ex,'decode_operation',side_effect=tracked):
   q.status(self.root);q.tick(self.root);q.seal(self.root,allow_empty=True);q.finalize(self.root)
  self.assertEqual(len(seen),3);self.assertTrue(all(m.stats['cleared']and not m.values for m in seen));self.assertIsNone(ex._DECODE_MEMO.get())
 def test_operation_error_clears_scope_and_releases_D(self):
  original=ex.decode_operation;seen=[]
  @contextmanager
  def tracked():
   with original()as m:seen.append(m);yield m
  with patch.object(ex,'decode_operation',side_effect=tracked),patch.object(q.finite_intake,'frontier',side_effect=OSError('invented')):
   with self.assertRaises(OSError):q.tick(self.root)
  self.assertTrue(seen[0].stats['cleared']);self.assertIsNone(ex._DECODE_MEMO.get());_,fd=q.dispatcher(self.root);os.close(fd)
if __name__=='__main__':unittest.main(verbosity=2)
