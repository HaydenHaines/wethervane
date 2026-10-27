"""Append-only experiment recorder, not a scheduler or scientific verifier.

Decisions record recommendations only; no qualification or active-model pointer exists.
Artifact copies can survive a failed database transaction and are deliberately retained.
SQLite and local filesystem durability are assumed; external file tampering is detected
on artifact reads, not prevented. There is no backup, scheduler, scientific stage-schema
validation, or registry integration. UTC times describe recording, not experiment dates.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("research", "planned", "developing", "evaluating", "reviewing", "decided")
REFERENCES = ("plan", "evaluation", "review")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class LedgerError(ValueError):
    pass


class Ledger:
    def __init__(self, root):
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "ledger.sqlite3", isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS cycles(id TEXT PRIMARY KEY, binding TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS artifacts(hash TEXT PRIMARY KEY, size INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS checkpoints(
          cycle TEXT NOT NULL REFERENCES cycles(id), revision INTEGER NOT NULL,
          status TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY(cycle, revision));
        CREATE TABLE IF NOT EXISTS commands(id TEXT PRIMARY KEY, payload TEXT NOT NULL, response TEXT NOT NULL);
        """)
        for table in ("cycles", "artifacts", "checkpoints", "commands"):
            for operation in ("UPDATE", "DELETE"):
                self.db.execute(f"""CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation}
                  BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'append-only'); END""")

        # BEFORE INSERT also rejects REPLACE independently of recursive_triggers.
        for table, keys in {
            "cycles": ("id",),
            "artifacts": ("hash",),
            "checkpoints": ("cycle", "revision"),
            "commands": ("id",),
        }.items():
            predicate = " AND ".join(f"{key}=NEW.{key}" for key in keys)
            self.db.execute(f"""CREATE TRIGGER IF NOT EXISTS immutable_{table}_INSERT
              BEFORE INSERT ON {table} WHEN EXISTS(SELECT 1 FROM {table} WHERE {predicate})
              BEGIN SELECT RAISE(ABORT, 'append-only duplicate'); END""")

    def close(self):
        self.db.close()

    def _command(self, command_id, payload, action):
        if not isinstance(command_id, str) or not command_id.strip():
            raise LedgerError("command_id is required")
        encoded = canonical(payload)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            previous = self.db.execute("SELECT * FROM commands WHERE id=?", (command_id,)).fetchone()
            if previous:
                if previous["payload"] != encoded:
                    raise LedgerError("command_id reused with different payload")
                result = json.loads(previous["response"])
            else:
                result = action()
                self.db.execute("INSERT INTO commands VALUES(?,?,?)", (command_id, encoded, canonical(result)))
            self.db.execute("COMMIT")
            return result
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def create_cycle(self, command_id, cycle_id, incumbent_config, scope, target_layer, hypothesis):
        if not all(isinstance(x, str) and x.strip() for x in (cycle_id, target_layer, hypothesis)):
            raise LedgerError("cycle, target layer and hypothesis are required")
        if not isinstance(incumbent_config, dict) or not incumbent_config or not isinstance(scope, dict) or not scope:
            raise LedgerError("full incumbent configuration and scope must be nonempty objects")
        binding = dict(incumbent_config=incumbent_config, scope=scope, target_layer=target_layer, hypothesis=hypothesis)

        def action():
            self.db.execute("INSERT INTO cycles VALUES(?,?)", (cycle_id, canonical(binding)))
            body = dict(
                outcome="success",
                note="Cycle recorded",
                references={},
                recommendation=None,
                recorded_at=datetime.now(timezone.utc).isoformat(),
            )
            self.db.execute("INSERT INTO checkpoints VALUES(?,?,?,?)", (cycle_id, 0, "research", canonical(body)))
            return dict(cycle=cycle_id, revision=0, status="research")

        return self._command(command_id, dict(operation="create", cycle=cycle_id, binding=binding), action)

    def _artifact(self, digest):
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise LedgerError("invalid artifact hash")
        row = self.db.execute("SELECT size FROM artifacts WHERE hash=?", (digest,)).fetchone()
        path = self.objects / digest
        if not row or path.is_symlink() or not path.is_file():
            raise LedgerError("missing artifact")
        data = path.read_bytes()
        if len(data) != row["size"] or hashlib.sha256(data).hexdigest() != digest:
            raise LedgerError("artifact corruption")
        return data

    def put_artifact(self, command_id, source):
        data = Path(source).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        destination = self.objects / digest
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or destination.read_bytes() != data:
                raise LedgerError("artifact corruption")
        else:
            name = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.objects, delete=False) as temporary:
                    name = temporary.name
                    temporary.write(data)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                try:
                    os.link(name, destination)
                    destination.chmod(0o444)
                except FileExistsError:
                    if destination.is_symlink() or destination.read_bytes() != data:
                        raise LedgerError("artifact corruption")
            finally:
                if name is not None:
                    os.unlink(name)
        # Sync even existing objects: another publisher may have linked without
        # completing its directory flush. Registration must not outrun durability.
        descriptor = os.open(self.objects, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        def action():
            if not self.db.execute("SELECT hash FROM artifacts WHERE hash=?", (digest,)).fetchone():
                self.db.execute("INSERT INTO artifacts VALUES(?,?)", (digest, len(data)))
            self._artifact(digest)
            return dict(sha256=digest, size=len(data))

        result = self._command(command_id, dict(operation="artifact", sha256=digest, size=len(data)), action)
        self._artifact(digest)
        return result

    def status(self, cycle_id):
        row = self.db.execute(
            "SELECT * FROM checkpoints WHERE cycle=? ORDER BY revision DESC LIMIT 1", (cycle_id,)
        ).fetchone()
        if not row:
            raise LedgerError("unknown cycle")
        return dict(cycle=cycle_id, revision=row["revision"], status=row["status"])

    def checkpoint(
        self,
        command_id,
        cycle_id,
        expected_revision,
        status,
        note,
        references=None,
        outcome="success",
        recommendation=None,
    ):
        references = {} if references is None else references
        if status not in STATUSES or outcome not in ("success", "failure"):
            raise LedgerError("invalid status or outcome")
        if type(expected_revision) is not int or expected_revision < 0 or not isinstance(note, str) or not note.strip():
            raise LedgerError("revision and note required")
        if not isinstance(references, dict) or set(references) - set(REFERENCES):
            raise LedgerError("unknown reference role")
        if recommendation not in (None, "retain", "revise", "reject", "defer", "promote"):
            raise LedgerError("invalid recommendation")
        if (status == "decided") != (recommendation is not None):
            raise LedgerError("decided checkpoint requires recommendation; recommendations belong to decisions")
        body = dict(note=note, references=references, outcome=outcome, recommendation=recommendation)
        payload = dict(
            operation="checkpoint", cycle=cycle_id, expected_revision=expected_revision, status=status, body=body
        )

        def action():
            current = self.status(cycle_id)
            if current["revision"] != expected_revision:
                raise LedgerError("stale revision")
            if outcome == "failure" and status != current["status"]:
                raise LedgerError("failure cannot advance phase")
            if STATUSES.index(status) < STATUSES.index(current["status"]):
                raise LedgerError("phase cannot move backward; record a new cycle")
            for digest in references.values():
                self._artifact(digest)
            revision = expected_revision + 1
            recorded_body = dict(body, recorded_at=datetime.now(timezone.utc).isoformat())
            self.db.execute(
                "INSERT INTO checkpoints VALUES(?,?,?,?)", (cycle_id, revision, status, canonical(recorded_body))
            )
            return dict(cycle=cycle_id, revision=revision, status=status)

        return self._command(command_id, payload, action)

    def export(self, cycle_id):
        row = self.db.execute("SELECT binding FROM cycles WHERE id=?", (cycle_id,)).fetchone()
        if not row:
            raise LedgerError("unknown cycle")
        checkpoints = []
        for item in self.db.execute("SELECT * FROM checkpoints WHERE cycle=? ORDER BY revision", (cycle_id,)):
            body = json.loads(item["body"])
            for digest in body["references"].values():
                self._artifact(digest)
            checkpoints.append(dict(revision=item["revision"], status=item["status"], **body))
        return dict(
            schema_version=1,
            recorder_only=True,
            cycle=cycle_id,
            binding=json.loads(row["binding"]),
            checkpoints=checkpoints,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    sub = parser.add_subparsers(dest="operation", required=True)
    create = sub.add_parser("create")
    create.add_argument("command_id")
    create.add_argument("cycle")
    create.add_argument("binding", help="JSON file: incumbent_config, scope, target_layer, hypothesis")
    artifact = sub.add_parser("artifact")
    artifact.add_argument("command_id")
    artifact.add_argument("path")
    checkpoint = sub.add_parser("checkpoint")
    checkpoint.add_argument("command_id")
    checkpoint.add_argument("cycle")
    checkpoint.add_argument("expected_revision", type=int)
    checkpoint.add_argument("status", choices=STATUSES)
    checkpoint.add_argument("note")
    checkpoint.add_argument("--references", default="{}", help="JSON object of plan/evaluation/review artifact hashes")
    checkpoint.add_argument("--outcome", choices=("success", "failure"), default="success")
    checkpoint.add_argument("--recommendation", choices=("retain", "revise", "reject", "defer", "promote"))
    for operation in ("status", "export"):
        sub.add_parser(operation).add_argument("cycle")
    args = parser.parse_args()
    ledger = Ledger(args.root)
    try:
        if args.operation == "create":
            result = ledger.create_cycle(args.command_id, args.cycle, **json.loads(Path(args.binding).read_text()))
        elif args.operation == "artifact":
            result = ledger.put_artifact(args.command_id, args.path)
        elif args.operation == "checkpoint":
            result = ledger.checkpoint(
                args.command_id,
                args.cycle,
                args.expected_revision,
                args.status,
                args.note,
                json.loads(args.references),
                args.outcome,
                args.recommendation,
            )
        else:
            result = getattr(ledger, args.operation)(args.cycle)
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        ledger.close()


if __name__ == "__main__":
    main()
