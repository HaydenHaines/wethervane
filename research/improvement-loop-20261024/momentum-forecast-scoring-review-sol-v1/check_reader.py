"""Independent invented-file admission/label boundary; no actual forecast values."""
from pathlib import Path
from unittest.mock import patch
import sys,tempfile,json
HERE=Path(__file__).resolve().parent;P=HERE.parent/'momentum-forecast-scoring-sol-v1'
sys.path[:0]=[str(P),str(P/'snapshots')]
import score_reader as r
import synthetic_files as f
opens=[]
original=r.checked_bytes
with tempfile.TemporaryDirectory() as tmp:
 pin,admission,fixture=f.make_bundle(Path(tmp))
 def spy(pin):
  if str(pin['path']).endswith('SYNTHETIC-outcomes.csv'):opens.append('numeric_synthetic_label_bytes')
  return original(pin)
 with patch.object(r,'checked_bytes',spy):
  verified=r.verify_admission(pin)
  assert not opens
  try:r.read_labels_after_admission({'audit':['all_prediction_reference_mask_status_receipts_verified_before_numeric_label_read']})
  except r.ArtifactHardFailure:pass
  else:raise AssertionError('forged marker accepted')
  assert not opens
  try:verified['folds'][2018]['mask']['common_polled_race_ids']=()
  except TypeError:pass
  else:raise AssertionError('mutable verified mask')
  try:verified['folds'][2018]['jobs']['FS5']['means'].setflags(write=True)
  except ValueError:pass
  else:raise AssertionError('reversible verified array')
  result=r.read_labels_after_admission(verified)
  assert opens==['numeric_synthetic_label_bytes']
  assert len(result['labels'])==5
  assert result['frequency_references'][2018]['probability']==.5
  assert len(verified['folds'][2018]['mask']['near_even_race_ids'])==0
out=dict(schema='IndependentSyntheticScoringReaderCheckSolV1',synthetic_files_only=True,numeric_label_open_after_verified_receipts=True,forged_marker_rejected=True,immutable_mask_and_arrays=True,earlier_frequency_scope_verified=True,foreign_nonnumeric_rows_filtered_before_numeric_parse=True,empty_near_even_preserved=True,actual_target_labels=False,actual_forecasts=False)
(HERE/'reader-checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
