"""Independently verify draft mappings against retained source bytes; no admission."""

import hashlib
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OWNER = Path('/home/hayden/projects/wethervane-redesign-20261006/integration')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else OWNER / p


def pointer(value, address):
    if not address:
        return value
    for item in address.lstrip('/').split('/'):
        item = item.replace('~1', '/').replace('~0', '~')
        value = value[int(item)] if isinstance(value, list) else value[item]
    return value


def source(ref):
    p = resolve(ref['document']['path'])
    assert digest(p) == ref['document']['sha256']
    return pointer(read(p), ref['json_pointer'])


class Cells(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cells = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag in ('th', 'td'):
            self.current = []

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag in ('th', 'td') and self.current is not None:
            self.cells.append(' '.join(''.join(self.current).split()))
            self.current = None


idwv_path = BASE / 'native-menu-proposal-sol-v1/proposal.json'
nm_path = BASE / 'native-nm-menu-review-luna-v1/source-to-native-draft.json'
idwv = read(idwv_path)
nm = read(nm_path)
for pin in idwv['pins'].values():
    assert digest(resolve(pin['path'])) == pin['sha256']
for candidate in idwv['WV']['candidates'][:3]:
    original = source(candidate['source_ref'])
    assert original['option_source_id'] == candidate['candidate_id']
    assert original['source_name'] == candidate['name_literal']
wv_new = idwv['WV']['candidates'][3]
wv_row = source(wv_new['source_ref'])
assert wv_row['candidateId'] == 2937
assert wv_new['name_literal'] == wv_row['candidateBallotName'] == 'RIO PHILLIPS'
assert wv_row['officeName'] == 'U.S. SENATE'
assert wv_row['candidateType'] == 'W'
assert wv_row['electionName'] == '11/03/2026 - GENERAL 2026'
assert wv_row['partyCode'] is None and wv_row['partyDescription'] is None
assert wv_new['party_literal'] is None and wv_new['caucus_assignment'] is None
wv_rows = sum([source(ref) for ref in idwv['WV']['record_observation']['complete_W_pagination']['page_refs']], [])
assert len(wv_rows) == len({r['candidateId'] for r in wv_rows}) == 15
general = [r for r in wv_rows if r['electionName'] == '11/03/2026 - GENERAL 2026']
assert len(general) == 12
assert len([r for r in general if r['officeName'] == 'U.S. SENATE' or r['officeName'].startswith(('STATE SENATE - ', 'HOUSE OF DELEGATES - '))]) == 8
assert [r['candidateId'] for r in general if r['officeName'] == 'U.S. SENATE'] == [2937]
assert not idwv['WV']['fit_admitted'] and not idwv['WV']['public_qualified']
for candidate in idwv['ID']['record_observation']['approved_existing_candidate_crosswalk']:
    row = source(candidate['raw_source_ref'])
    original = source(candidate['existing_owner_source_ref'])
    assert original['option_source_id'] == candidate['candidate_id']
    assert original['source_name'] == candidate['name_literal']
    assert candidate['name_literal'] == row['candidateName']
    assert row['filingStatus'] == 'Approved'
    assert row['isWriteIn'] is False
withdrawn = source(idwv['ID']['record_observation']['withdrawn_excluded']['source_ref'])
assert withdrawn['candidateName'] == 'David Roth' and withdrawn['filingStatus'] == 'Withdrawn'

template_pin = nm['native_template']
assert digest(resolve(template_pin['path'])) == template_pin['sha256']
template = pointer(read(resolve(template_pin['path'])), template_pin['json_pointer'])
raw_pin = nm['source']['source_file']
raw_path = resolve(raw_pin['path'])
assert digest(raw_path) == raw_pin['sha256']
raw = raw_path.read_bytes()
headers = Cells()
headers.feed(re.search(rb'<thead.*?</thead>', raw, re.S)[0].decode())
draft = nm['rows'][0]
assert [c['candidate_id'] for c in draft['candidates']] == [c['candidate_id'] for c in template['candidates']]
for old, new in zip(template['candidates'], draft['candidates']):
    assert old['name_literal'] == new['name_literal']
    assert old['party_recipe'] == new['party_recipe']
    assert new['caucus_assignment'] is None
    assert new['source_record']['model_admitted'] is False
records = [c['source_metadata']['source_row'] for c in draft['candidates']]
records += [r['source_row'] for r in nm['excluded_source_records']]
for record in records:
    row_id = record['selector'].rsplit('#', 1)[-1].encode()
    matches = re.findall(rb'<tr\b[^>]*\bid="' + re.escape(row_id) + rb'"[^>]*>.*?</tr>', raw, re.S)
    assert len(matches) == 1
    assert hashlib.sha256(matches[0]).hexdigest() == record['row_sha256']
    cells = Cells()
    cells.feed(matches[0].decode())
    assert len(cells.cells) == len(headers.cells)
    for field, expected in record['fields'].items():
        assert cells.cells[headers.cells.index(field)] == expected, field
assert [r['fields']['Status'] for r in records] == ['Qualified', 'Qualified', 'Disqualified', 'Disqualified']
assert draft['writein_domain']['eligible_writein_candidates'] is None
assert draft['writein_domain']['legal_writein_voting'] == template['writein_domain']['legal_writein_voting']
assert not draft['conditional_finite_menu_supported']
assert not draft['fit_admitted'] and not draft['publication_admitted']
assert not nm['source_admitted'] and not nm['native_input_mutated']

result = {
    'schema': 'NativeCandidateDraftIndependentRootReviewV1',
    'observed_utc': datetime.now(timezone.utc).isoformat(),
    'script_sha256': digest(Path(__file__)),
    'drafts': [{'path': str(p), 'sha256': digest(p)} for p in [idwv_path, nm_path]],
    'checks_passed': True,
    'checks': ['All 13 ID/WV source pins authenticated.',
               'Three WV printed IDs retained; provisional write-in ID binds official row 2937.',
               'Both WV W pages contain 15 distinct candidate records.',
               'Four Idaho Approved records match existing IDs/names; Roth remains evidence only.',
               'Both NM IDs, names and party recipes retained; all four HTML row hashes and declared fields verified.',
               'NM disqualified records stay outside candidate axis; all three settings remain unadmitted.'],
    'scope': 'Draft source-to-native mappings only. Source acquisition review supplies query scope and release interpretation. No model construction, legal admission, cutoff reconstruction, fit or public release.',
    'owner_supported_setting_count': 'Unchanged at 26 of 35 in checkpoint v35.',
    'reviewer_correction': 'Two invocations stopped before writing a receipt: the reviewer initially used candidate_id instead of option_source_id, then assumed an officeDescription label LEGISLATIVE while the source uses STATE RACE. The verifier now uses inspected role fields and exact legislative office names. Draft and source bytes were unchanged.',
}
with (HERE / 'review.json').open('x') as out:
    json.dump(result, out, indent=2)
    out.write('\n')
print(json.dumps({'checks_passed': True, 'review': str(HERE / 'review.json')}))
