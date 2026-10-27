"""Two tiny synthetic boundary probes against the pinned same-UID reference."""
import hashlib, importlib.util, json, shutil, tempfile
from pathlib import Path
SOURCE = Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/queue-history-authority-reference-sol-v1/reference.py')
EXPECTED = 'dbfb14d9b9365fcf0a9feaaebe24b11f80e874273b601ea55606a8a89027f1f2'
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED
with tempfile.TemporaryDirectory(prefix='history-authority-probe-') as tmp:
    root = Path(tmp)
    module_path = root / 'reference.py'
    module_path.write_bytes(raw)
    spec = importlib.util.spec_from_file_location('pinned_reference', module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.Reference.create(root / 'domain')
    status = r.status(); genesis = status['head']; lease = status['lease']
    first = r.admit('same-key', genesis, lease, 'first', '/out/first')
    status = r.status(); r.advance_lease('lease-2', status['head'], status['lease'])
    status = r.status(); r.admit('second-key', status['head'], status['lease'], 'second', '/out/second')
    head_before = r.canonical_head()[0]
    replay = r.admit('same-key', genesis, lease, 'first', '/out/first')
    assert replay == first and replay['historical_commit_only'] is True and replay['live_launch_authorization'] is False
    assert r.canonical_head()[0] == head_before
    try:
        r.advance_lease('same-key', genesis, lease)
    except mod.Refused as exc:
        assert str(exc) == 'idempotency_body_conflict'
    else:
        raise AssertionError('same key reused for another action')
    forged = {
        'domain': genesis['domain'],
        'head': {'domain': genesis['domain'], 'sequence': genesis['sequence'] + 1, 'hash': 'f' * 64},
        'lease': 1, 'nonce': 'n', 'floor_checked': genesis, 'response_authenticated': False,
    }
    reply = mod.check_response(forged, 'n', genesis['domain'], genesis)
    assert reply['consistent_reply_claim'] and reply['authenticated'] is False
    # The active cleanup fields are only a same-UID caller assertion in this reference.
    r2 = mod.Reference.create(root / 'domain2')
    s = r2.status(); r2.admit('a', s['head'], s['lease'], 'a', '/out/a')
    s = r2.status(); r2.reserve('reserve-a', s['head'], s['lease'], 'a', 'invented-worker')
    s = r2.status(); active = s['active']
    fake_cleanup = {
      'schema':'InventedCleanupEvidenceV1','job':active['job'],'worker_token':active['worker_token'],
      'reservation_hash':active['reservation_hash'],'worker_status':'exited','cleanup_verified':True,
      'outcome':'accepted_unqualified','terminal_sha256':'a'*64,'cleanup_sha256':'b'*64,
      'scientifically_qualified':False,
    }
    s = r2.status(); receipt = r2.release('release-a', s['head'], s['lease'], fake_cleanup)
    assert receipt['runtime_worker_evidence_authenticated'] is False
    print(json.dumps({
      'schema':'HistoryAuthorityTargetedProbeV1',
      'pinned_source_sha256':EXPECTED,
      'exact_retry_after_later_head_and_lease':'returns original historical receipt; does not append or authorize launch',
      'same_key_different_action':'refused with idempotency_body_conflict',
      'echoed_floor_forged_later_head':'accepted only as unauthenticated consistency claim; no ancestry proof',
      'coherent_fake_cleanup':'accepted by synthetic reference, receipt explicitly unauthenticated',
      'head_after_retry':head_before,
      'trust':'trusted_mutable_same_uid_reference'
    }, sort_keys=True))
