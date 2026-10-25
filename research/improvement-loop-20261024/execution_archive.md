# Portable completed-attempt execution evidence

`execution_archive.py` exports a completed local execution attempt into a fresh,
read-only evidence package. It does not restore a store, adopt a process, retry a
job, execute archived code, interpret scientific diagnostics, or change
`qualification: "unassessed"`. The original `execution.inspect` remains strict
about the current job-lock device and inode for live ownership. A copied lock
inode is historical metadata, never evidence that a process is owned now.

## Contract

Export requires the selected attempt and every recursively referenced
`previous_attempt` to have a verified terminal/recovery record, no live recorded
process or session member, and no held cooperating job lock. The shared job lock
can conservatively block export of an older completed attempt while its retry
is running. Export authenticates the current source store and all declared pins;
missing or changed source bytes fail closed. It never modifies the source store.

The fresh destination contains `manifest.json` and `objects/sha256/<digest>`.
It must be outside the source store and native output directories.
Each object is streamed from an authenticated regular source file and checked
against its declared SHA256 and size. The exporter verifies the complete copied
object inventory and graph, then syncs object files and directory entries before
atomically publishing the manifest last. It maps each
original absolute path and pin to an object, and binds the ordered attempt chain.
The exact original specification, prepared, terminal/recovery, claim, native
receipt and lifecycle bytes are preserved; JSON references are not rewritten.
The export response gives the manifest SHA256 and size. Keep that pin outside
the archive, such as in an independent review record. The archive cannot
authenticate its own manifest without an external expected digest.

The traversal is deliberately finite:

- Executor specification, prepared, terminal/recovery, job claim, output claim,
  lifecycle evidence and declared native receipt pins.
- Each specification's previous-attempt receipt and its completed predecessor
  chain in the same cycle/job.
- Explicit input/source pins, resolved invoked executable pin, manager source,
  manager interpreter target and manager dependency pins.

Native scientific receipt bytes are opaque except for the exact top-level
metadata declared by `native_receipts`. No files referenced only *inside* a
native receipt are followed. The package does not prove downstream scientific
lineage, recreate an executable symlink/working environment, or retrieve missing
files from Git, the network, or a package manager. Large declared inputs and
runtime files may make the archive large; copying is streamed. Parsed executor
and native JSON documents are capped at 16 MiB; other pinned bytes have no
document parse cap.

## CLI and API

```sh
python3 -B research/improvement-loop-20261024/execution_archive.py export \
  --root STORE --attempt ATTEMPT --destination FRESH_ARCHIVE
python3 -B research/improvement-loop-20261024/execution_archive.py inspect \
  --archive MOVED_ARCHIVE --manifest-sha256 EXTERNALLY_RETAINED_SHA256
```

The equivalent API is `export_completed_attempt(root, attempt_id, destination)`
and `inspect_archive(directory, manifest_sha256, manifest_size=None)`.
Inspection reads only the archive, checks the external manifest pin, streams
and hashes every object, validates the original executor reference graph and
native declared metadata, and reports `qualification: "unassessed"`. It does
not consult original absolute source paths, `/proc`, or the copied lock inode.
The package may be moved and inspected after the source store and owner inputs
are gone. An incomplete export has no manifest and fails inspection. The
destination is never overwritten.

## Verification

```sh
python3 -B -m unittest discover -s research/improvement-loop-20261024 \
  -p test_execution_archive.py -v
```

Tests run only tiny invented subprocesses. They cover a completed retry chain,
relocation and source removal, external manifest pinning, object corruption or
absence, conflicting pin maps, live export refusal, and no source mutation or
destination overwrite.
