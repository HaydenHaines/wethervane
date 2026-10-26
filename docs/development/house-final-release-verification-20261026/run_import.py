"""Root-owned, one-release admission and execution record. Does not fit models."""
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
import time

BASE = pathlib.Path('/home/hayden/projects/ballotmancy-house-benchmark/research/house-live-20261025')
NATIVE = BASE / 'release-v6/native'
SITE = pathlib.Path('/home/hayden/projects/ballotmancy-house-live-ui-20261025')
LAB = pathlib.Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024')
OUT = pathlib.Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readpin(path, expected):
    assert sha(path) == expected, str(path)
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


assert not (OUT / 'import-execution-v1.json').exists(), 'Do not reuse this execution record.'
assert not (SITE / 'data/house-live').exists(), 'Do not replace an installed release.'
pins = json.loads((LAB / 'house-final-release-review-sol-v1/pins.json').read_text())
for relative, expected in pins['metadata_pins'].items():
    assert sha(BASE / relative) == expected, relative
catalog = readpin(NATIVE / 'catalog.json', 'b602e1d3202df89eb058a71dcbc564b60755a01fc22902e2a39b2690e1df2368')
results = [readpin(NATIVE / row['result']['path'], row['result']['sha256']) for row in catalog['entries']]
qualification = readpin(NATIVE / catalog['qualification']['path'], catalog['qualification']['sha256'])
geo = results[0]['geography']
geometry = readpin(NATIVE / geo['path'], geo['sha256'])
readpin(NATIVE / geo['source_receipt']['path'], geo['source_receipt']['sha256'])
district_ids = {feature['properties']['districtKey'] for feature in geometry['features']}
assert len(district_ids) == 435
roster = [(r['race_id'], r['district_key'], r['candidates']) for r in results[0]['races']]
for result in results:
    assert {r['district_key'] for r in result['races']} == district_ids
    assert [(r['race_id'], r['district_key'], r['candidates']) for r in result['races']] == roster
    assert all(note in result['qualification']['limitations'] for note in qualification['limitations'])
assert sum(r['joint_draws']['count'] for r in results) == 26_000_000
poll_root = BASE / 'polls-v3'
poll_receipt = '13127f2839a2e65e91f66c1020ea00c966787dc3aba79057f342d769ccffa7a2'
poll_ledger = '67eedef5f40f3b233a0e1f557cface5e430e6f64cef3fa59d380f725ea6c3c73'
admission = json.loads((NATIVE / 'evidence/admission-v1.json').read_text())
for name, expected in [('receipt.json', poll_receipt), ('ledger.json', poll_ledger)]:
    readpin(poll_root / name, expected)
    assert any(p['path'].endswith('/polls-v3/' + name) and p['sha256'] == expected for p in admission['input_pins'])

review = {
    'schema': 'HouseLiveSiteReviewV1', 'decision': 'publish',
    'joint_format': 'house_live_npz_chain_major_v1', 'seat_quantile_convention': 'inverse_cdf',
    'catalog_sha256': sha(NATIVE / 'catalog.json'),
    'source_policy_sha256': catalog['source_policy']['sha256'],
    'qualification_sha256': catalog['qualification']['sha256'],
    'geography_sha256': geo['sha256'],
    'geography_source_receipt_sha256': geo['source_receipt']['sha256'],
    'results': [{'id': e['id'], 'result_sha256': e['result']['sha256'],
                 'joint_draws_sha256': r['joint_draws']['sha256']} for e, r in zip(catalog['entries'], results)],
    'checks': dict.fromkeys(['inputsVerified', 'calculationsReviewed', 'jointRowsVerified',
                            'crossModelRosterVerified', 'geographyVerified'], True),
    'limitations': qualification['limitations'],
}
review_path = OUT / 'operator-review.json'
write(review_path, review)
write(OUT / 'operator-review-provenance.json', {
    'schema': 'HouseSiteOperatorReviewProvenanceV1',
    'createdAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status': 'operator_admission_for_offline_import_website_checks_still_required',
    'reviewSha256': sha(review_path),
    'scientificReviewSha256': sha(BASE / 'release-v6/scientific-review.json'),
    'nativeReviewSha256': sha(BASE / 'release-v6/native-review.json'),
    'independentReviewSha256': sha(LAB / 'house-final-release-review-sol-v1/review.md'),
    'independentPinsSha256': sha(LAB / 'house-final-release-review-sol-v1/pins.json'),
    'jointRowsVerifiedBasis': 'Authenticated independent scientific raw-array audits and native export reconciliation already passed for all16. Root importer additionally streams every26M joint winner row before atomically installing any publication. This admission does not assert that the forthcoming root importer or browser checks have completed.',
    'rootCompactChecks': {'metadataPinsMatched': len(pins['metadata_pins']), 'resultsMatched': 16,
                         'districtsAndMapIDs': 435, 'sameCandidateRoster': True,
                         'originalPollingPinsBoundToAdmission': True},
    'publicationGatesRemaining': ['root full native archive import', 'real data desktop/mobile tests',
                                 'deployment and served verification'],
})
command = [
    'node', '--experimental-strip-types', 'scripts/import_house_live.mjs',
    '--source-root', str(NATIVE), '--catalog', 'catalog.json', '--catalog-sha256', sha(NATIVE / 'catalog.json'),
    '--review-root', str(OUT), '--review', 'operator-review.json', '--review-sha256', sha(review_path),
    '--geography-source-name', 'U.S. Census Bureau TIGER/Line congressional districts',
    '--geography-source-url', 'https://www2.census.gov/geo/tiger/TIGER2026/CD/',
    '--geography-source-note', '2026 CD120 boundaries except the reviewed Missouri 2024 CD119 correction; shared borders simplified at 200 meters. Exact source URLs and geometry review remain in the source receipt.',
    '--output-dir', str(SITE / 'data/house-live'), '--python', 'python3',
    '--polling-root', str(poll_root), '--polling-receipt', 'receipt.json', '--polling-receipt-sha256', poll_receipt,
    '--polling-ledger', 'ledger.json', '--polling-ledger-sha256', poll_ledger,
]
receipt = {'schema': 'HouseFinalNativeImportExecutionV1', 'command': command,
           'startedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'reviewSha256': sha(review_path), 'catalogSha256': sha(NATIVE / 'catalog.json'),
           'runnerSha256': sha(pathlib.Path(__file__)), 'status': 'running'}
receipt_path = OUT / 'import-execution-v1.json'
write(receipt_path, receipt)
print(json.dumps({'status': 'starting', 'receipt': str(receipt_path)}), flush=True)
started = time.monotonic()
with (OUT / 'import-attempt1.log').open('w') as log:
    process = subprocess.Popen(command, cwd=SITE, stdout=log, stderr=subprocess.STDOUT)
    receipt['processPID'] = process.pid
    write(receipt_path, receipt)
    code = process.wait()
receipt.update(status='passed' if code == 0 else 'failed', returnCode=code,
               elapsedSeconds=time.monotonic() - started,
               finishedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               logSha256=sha(OUT / 'import-attempt1.log'))
write(receipt_path, receipt)
print(json.dumps(receipt, indent=2))
print((OUT / 'import-attempt1.log').read_text()[-7000:])
sys.exit(code)
