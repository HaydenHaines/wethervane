# Original momentum source document storage

The event-identity packet's original PDFs and HTML remain at `momentum-event-identity-intake-luna-v1/raw/` locally. All eleven documents are also copied into the content journal by SHA-256 and included in the fully restored-and-verified [v6 backup](checkpoints/research-backup-v6.json). Git stores the derived evidence, source URLs and exact manifests; it does not distribute these original documents.

After restoring that backup to a separate directory, each missing document can be recovered from `objects/<sha256>` using its `raw/...` destination and hash in [the source inventory](momentum-event-identity-intake-luna-v1/source-inventory.json). Check the recovered bytes against the manifest before use. The source URLs offer an acquisition route but do not guarantee that a later download has the frozen bytes. No local original has been removed.

The backup is local; it is not an external recovery service. Later calibration execution code and results are outside this already-created v6 snapshot and require subsequent preservation.
