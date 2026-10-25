"""One-time G01 journal recovery snapshot; never fits or changes model results."""

import hashlib
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/hayden/projects/ballotmancy-improvement-lab')
BASE = ROOT / 'research/improvement-loop-20261024'
JOURNAL = ROOT / '.improvement-ledger'
BACKUPS = JOURNAL / 'backups'
DESTINATION = BACKUPS / 'research-checkpoint-v16-delta.zip'
RECEIPT = BASE / 'checkpoints/research-backup-v16-delta.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if DESTINATION.exists() or RECEIPT.exists():
        raise FileExistsError('Preserve existing v16 artifacts; no overwrite')
    previous = json.loads((BASE / 'checkpoints/research-backup-v15-delta.json').read_text())
    required = previous['required_archives'] + [{
        'name': Path(previous['local_backup']).name, 'sha256': previous['sha256'],
    }]
    old_objects = set()
    for item in required:
        archive = BACKUPS / item['name']
        assert digest(archive) == item['sha256'], archive
        with zipfile.ZipFile(archive) as source:
            for name in source.namelist():
                if name.startswith('objects/'):
                    assert re.fullmatch(r'objects/[0-9a-f]{64}', name), name
                    old_objects.add(name.removeprefix('objects/'))
    print(f'Verified {len(required)} prior archives, {len(old_objects)} objects', flush=True)
    sys.path.insert(0, str(BASE))
    from ledger import Ledger

    with tempfile.TemporaryDirectory(prefix='ballotmancy-g01-recovery-') as temporary:
        temporary = Path(temporary)
        snapshot = temporary / 'snapshot.sqlite3'
        with closing(sqlite3.connect(JOURNAL / 'ledger.sqlite3')) as source:
            with closing(sqlite3.connect(snapshot)) as target:
                source.backup(target)
        with closing(sqlite3.connect(snapshot)) as database:
            objects = dict(database.execute('SELECT hash, size FROM artifacts'))
            cycles = [row[0] for row in database.execute('SELECT id FROM cycles ORDER BY id')]
        assert old_objects <= objects.keys(), 'Append-only journal lost an old object'
        new_objects = sorted(objects.keys() - old_objects)
        for key in new_objects:
            source = JOURNAL / 'objects' / key
            assert source.stat().st_size == objects[key] and digest(source) == key, source
        metadata = {
            'required_archives': required, 'new_objects': len(new_objects),
            'total_objects': len(objects), 'cycles': cycles,
            'scope': 'Completed G01 gas experiment, source/code versions, reviews and decisions',
        }
        with zipfile.ZipFile(DESTINATION, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(snapshot, 'ledger.sqlite3')
            for key in new_objects:
                archive.write(JOURNAL / 'objects' / key, 'objects/' + key)
            archive.writestr('delta-metadata.json', json.dumps(metadata, indent=2) + '\n')
        print(f'Wrote v16 with {len(new_objects)} new objects; verifying full restore', flush=True)
        restored = temporary / 'restored'
        (restored / 'objects').mkdir(parents=True)
        archives = [BACKUPS / row['name'] for row in required] + [DESTINATION]
        for path in archives:
            with zipfile.ZipFile(path) as archive:
                # Only explicitly permitted members are materialized; no extractall.
                with archive.open('ledger.sqlite3') as source, (restored / 'ledger.sqlite3').open('wb') as target:
                    shutil.copyfileobj(source, target)
                for name in archive.namelist():
                    if not name.startswith('objects/'):
                        continue
                    assert re.fullmatch(r'objects/[0-9a-f]{64}', name), name
                    target = restored / name
                    if target.exists():
                        assert digest(target) == target.name, target
                        continue
                    with archive.open(name) as source, target.open('xb') as stream:
                        shutil.copyfileobj(source, stream)
        assert len(list((restored / 'objects').iterdir())) == len(objects)
        for key, size in objects.items():
            target = restored / 'objects' / key
            assert target.stat().st_size == size and digest(target) == key, target
        live, recovered = Ledger(JOURNAL), Ledger(restored)
        try:
            revisions = {}
            for cycle in cycles:
                assert live.export(cycle) == recovered.export(cycle), cycle
                revisions[cycle] = live.status(cycle)['revision']
        finally:
            live.close()
            recovered.close()
        result = {
            'schema': 'ResearchBackupDeltaReceiptV16',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'local_backup': str(DESTINATION.relative_to(ROOT)),
            'sha256': digest(DESTINATION), 'bytes': DESTINATION.stat().st_size,
            'required_archives': required, 'new_objects': len(new_objects),
            'total_objects_verified': len(objects), 'restored_cycle_revisions': revisions,
            'full_restore_and_all_object_hashes_verified': True,
            'all_cycle_exports_equal': True, 'sqlite_consistent_backup': True,
            'scope': metadata['scope'], 'external_backup_service': False,
            'later_receipt_continuation_not_recursively_included': True,
        }
        with RECEIPT.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
