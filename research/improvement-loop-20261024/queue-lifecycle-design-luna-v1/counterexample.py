"""Tiny stdlib crash-boundary counterexamples; no production queue is imported."""
import json
import os
import sqlite3
import tempfile
from pathlib import Path


def dbopen(path):
    db = sqlite3.connect(path, isolation_level=None)
    db.execute("PRAGMA synchronous=FULL")
    return db


def init(path):
    db = dbopen(path)
    db.executescript("""
      CREATE TABLE releases(attempt TEXT PRIMARY KEY, digest TEXT NOT NULL);
      CREATE TABLE pending(attempt TEXT PRIMARY KEY, remaining INTEGER NOT NULL, blocked TEXT);
      CREATE TABLE edges(producer TEXT NOT NULL, consumer TEXT NOT NULL, PRIMARY KEY(producer,consumer));
      INSERT INTO pending VALUES ('middle',1,NULL),('leaf',1,NULL);
      INSERT INTO edges VALUES ('producer','middle'),('middle','leaf');
    """)
    db.close()


def publish(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def release_tx(path, attempt, digest, crash_before_commit=False):
    db = dbopen(path)
    db.execute("BEGIN IMMEDIATE")
    db.execute("INSERT INTO releases VALUES (?,?)", (attempt, digest))
    db.execute("UPDATE pending SET remaining=remaining-1 WHERE attempt='middle'")
    if crash_before_commit:
        os._exit(73)
    db.execute("COMMIT")
    db.close()


def run():
    with tempfile.TemporaryDirectory(prefix="queue-lifecycle-luna-") as td:
        root = Path(td)
        dbpath = root / "q.sqlite"
        release_file = root / "release.json"
        init(dbpath)
        payload = {"attempt": "producer", "outcome": "accepted_execution"}
        publish(release_file, payload)
        digest = __import__("hashlib").sha256(release_file.read_bytes()).hexdigest()

        pid = os.fork()
        if pid == 0:
            release_tx(dbpath, "producer", digest, crash_before_commit=True)
        _, status = os.waitpid(pid, 0)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 73
        db = dbopen(dbpath)
        rows = db.execute("SELECT * FROM releases").fetchall()
        middle = db.execute("SELECT remaining,blocked FROM pending WHERE attempt='middle'").fetchone()
        ready = db.execute("SELECT count(*) FROM pending WHERE remaining=0 AND blocked IS NULL").fetchone()[0]
        pending = db.execute("SELECT count(*) FROM pending").fetchone()[0]
        assert release_file.exists() and rows == [] and middle == (1, None) and ready == 0

        # Safe replay: exact persisted file is authenticated, then the database transition commits.
        release_tx(dbpath, "producer", digest)
        db = dbopen(dbpath)
        assert db.execute("SELECT digest FROM releases WHERE attempt='producer'").fetchone() == (digest,)
        assert db.execute("SELECT remaining FROM pending WHERE attempt='middle'").fetchone() == (0,)
        assert db.execute("SELECT remaining FROM pending WHERE attempt='leaf'").fetchone() == (1,)
        ready_after = db.execute("SELECT count(*) FROM pending WHERE remaining=0 AND blocked IS NULL").fetchone()[0]
        db.close()

        # Empty runnable frontier is not a completion signal: unresolved downstream work remains.
        db = dbopen(dbpath)
        db.execute("UPDATE pending SET blocked='failed producer' WHERE attempt='middle'")
        summary = {
            "external_release_published_before_db_commit": True,
            "crash_exit": 73,
            "after_crash_db_release_rows": len(rows),
            "after_crash_middle_remaining": middle[0],
            "after_crash_ready_rows": ready,
            "after_crash_pending_rows": pending,
            "replay_ready_rows": ready_after,
            "downstream_leaf_remaining": db.execute("SELECT remaining FROM pending WHERE attempt='leaf'").fetchone()[0],
            "empty_ready_not_complete": db.execute("SELECT count(*) FROM pending WHERE remaining=0 AND blocked IS NULL").fetchone()[0] == 0
                and db.execute("SELECT count(*) FROM pending").fetchone()[0] > 0,
            "journal_mode": db.execute("PRAGMA journal_mode").fetchone()[0],
            "synchronous": db.execute("PRAGMA synchronous").fetchone()[0],
        }
        db.close()
        print(json.dumps(summary, sort_keys=True, indent=2))
        assert summary["empty_ready_not_complete"]
        assert summary["journal_mode"] == "delete" and summary["synchronous"] == 2


if __name__ == "__main__":
    run()
