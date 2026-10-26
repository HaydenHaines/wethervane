import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import scripts.export_election_night_core_reference as adapter
ROOT=Path(__file__).resolve().parents[1]
class CoreReferenceExportTests(unittest.TestCase):
 @unittest.skipUnless(adapter.SOURCE.is_dir(), "Pinned Senate research source is unavailable")
 def test_actual_qualified_two_banks_publication_and_source_preservation(self):
  value=adapter.build(adapter.SOURCE,ROOT)
  self.assertEqual(value,json.loads((ROOT/'public/data/election-night/core-references.json').read_text()))
  publication=json.loads((ROOT/'data/election-night-core-reference-publication.json').read_text())
  self.assertEqual(publication['sha256'],adapter.sha(ROOT/'public/data/election-night/core-references.json'))
  self.assertEqual([e['runId'] for e in value['entries']],['current-fundamentals','current-polling'])
  self.assertEqual(value['entries'][0]['reference']['cohort'],value['entries'][1]['reference']['cohort'])
  for arm,e in zip(['F','FP'],value['entries'],strict=True):
   self.assertEqual(e['reference'],adapter.read(adapter.SOURCE/adapter.BASE/f'fixed32-scorecard-reference-root-v2/{arm}/reference.json'))
   self.assertEqual(len(e['reference']['cohort']),32)
   self.assertFalse(e['reference']['draw_identity']['resampling']);self.assertFalse(e['reference']['draw_identity']['cross_bank_pairing'])
   self.assertEqual(e['sourcePins']['failedV1EvidenceSha256'],adapter.FAILED_V1_SHA)
 @unittest.skipUnless(adapter.SOURCE.is_dir(), "Pinned Senate research source is unavailable")
 def test_semantic_mutations_fail_even_behind_a_mocked_authenticated_read(self):
  original=adapter.read
  path=(adapter.SOURCE/adapter.BASE/'fixed32-scorecard-reference-root-v2/FP/reference.json').resolve()
  changes=[lambda r:r['draw_identity'].__setitem__('shape',[40000]),lambda r:r['draw_identity'].__setitem__('weight_per_row',1/100000),
   lambda r:r['draw_identity'].__setitem__('cross_bank_pairing',True),lambda r:r['per_race'].reverse(),lambda r:r['cohort'][0]['first_candidate'].__setitem__('id','wrong-person'),
   lambda r:r['distributions']['mean_absolute_margin_error_pp'].__setitem__('mean',999),lambda r:r.__setitem__('excluded_from_this_reference',[])]
  for change in changes:
   def changed(p):
    result=original(p)
    if p.resolve()==path: result=copy.deepcopy(result);change(result)
    return result
   with self.subTest(change=change),patch.object(adapter,'read',side_effect=changed):
    with self.assertRaises(adapter.CoreReferenceExportError):adapter.build(adapter.SOURCE,ROOT)
 def test_actual_hash_guard_and_safe_path_reject_changed_or_escaped_bytes(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);file=root/'evidence';file.write_bytes(b'invented');digest=adapter.sha(file)
   self.assertEqual(adapter.pinned(root,'evidence',digest),file)
   file.write_bytes(b'changed')
   with self.assertRaises(adapter.CoreReferenceExportError):adapter.pinned(root,'evidence',digest)
   with self.assertRaises(adapter.CoreReferenceExportError):adapter.pinned(root,'../escape',digest)
   with self.assertRaises(adapter.CoreReferenceExportError):adapter.pinned(root,str(file),digest)
   (root/'outside-link').symlink_to('/etc/passwd')
   with self.assertRaises(adapter.CoreReferenceExportError):adapter.pinned(root,'outside-link',digest)
 @unittest.skipUnless(adapter.SOURCE.is_dir(), "Pinned Senate research source is unavailable")
 def test_deterministic_temp_export_does_not_replace_existing35_reference(self):
  legacy=ROOT/'public/data/election-night/references.json';before=legacy.read_bytes()
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);out=root/'core.json';pub=root/'publication.json'
   adapter.export(adapter.SOURCE,ROOT,out,pub);first=out.read_bytes();adapter.export(adapter.SOURCE,ROOT,out,pub)
   self.assertEqual(out.read_bytes(),first);self.assertEqual(json.loads(pub.read_text())['sha256'],hashlib.sha256(first).hexdigest())
  self.assertEqual(legacy.read_bytes(),before)
if __name__=='__main__':unittest.main()
