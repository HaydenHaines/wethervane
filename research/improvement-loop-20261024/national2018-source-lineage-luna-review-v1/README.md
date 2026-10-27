# Independent challenge: national 2018 source-lineage follow-up

## Verdict

The narrow inventory claim is confirmed under local read/hash trust: the original attempt plan names 165 source entries; all 165 stored `source-snapshots` match the plan pins and all 165 current source files also match. The selected national-problem package has 11 files. Those same 11 pins appear in the earlier scientific unadmitted plan. The four files omitted from the preceding gamma-source report—`assemble.py`, `dependencies.py`, `inputs_v3.py`, and `reader.py`—are all present in the attempt plan, match their retained snapshot and current source bytes, match the earlier plan, and match separate retained copies in the new source-lineage packet.

This closes the specific hash inventory gap. It does not prove that these bytes were the exact in-memory module bytes at the time of every import, nor does it turn mutable local plans/snapshots into an immutable execution attestation. Source-level inventory completeness is also not a formal proof that arbitrary runtime/dynamic imports were impossible. The attempt’s `complete_banks_unqualified` receipt remains unqualified; nothing here verifies compiled graph equivalence, sampler results, arrays or a scientific conclusion.

## Independent reproduction

I ran this packet's `review.py` against the original plans, launch/receipt records, all 165 source snapshots, their current source paths, the earlier scientific plan and the new retained copies of the 11 package snapshots. It used only JSON and byte hashing. It did not import model/project modules, build a graph, read NPZ/checkpoint arrays or write to owner data. The script asserts:

- source plan = retained attempt plan = launch/receipt/sampling plan hash (`fd44e436…`);
- exactly 165 source pins and 165 snapshot files, with all 165 snapshot and current hashes matching;
- all 11 selected package files match the earlier scientific plan and retained copies;
- runtime inventory matches its plan; source-input snapshots match their declared pins.

Observed source bytes hashed: 19,599,145. Command: `python3 -B review.py` from this directory. The exact compact result is in `review.json`.

## Four-file correction and 11-file mapping

The four prior omissions have these hashes:

| File | SHA-256 |
|---|---|
| `assemble.py` | `e28e5f715051f4f3aab0323e403250e57d1334535415d3348312164ef1d4888b` |
| `dependencies.py` | `f36a835dc82fc02e8963be81eb27db6d0d538aab19d68b150e0dd7039f44b95f` |
| `inputs_v3.py` | `85adb960b982be3e5dd10e2cbd29521a209ad380e22e323ad97646c523a96180` |
| `reader.py` | `70f9fda6564f5890d8bc9c80bb782d6e3d0762b4b454e2e53d010fe3db887b95` |

All four and the other seven selected package files pass plan/snapshot/current and earlier-plan pin checks. Full per-file entries are preserved in Sol's pinned `result.json`; this review's independent script rechecks all 165 source triples and specifically rehashes the new retained copies for the four omissions.

## Boundaries and hashes

Reviewed Sol packet pins: `README.md` `b630405c…`, `result.json` `73129aedd9c7113bafaf1abeb76e17fa483939c9bb5ae5fcd3a217a3d11513fc`, `evidence.json` `8ea67d6f173791e272ef7544f82da3594c73a40e2c9157905b84568efabb1502`, and `check.py` `f351d5eebd1ec2d8fd6576a681a75e53442afd08a742c318f4942fade16cf99e`. These pin the packet reviewed; independent outputs and local script have their own hashes in `evidence.json`.

The four previously omitted files are now evidenced, but source-byte matching alone does not establish which bytes Python interpreted if mutable files could change between check, import, and snapshot operations. The report's own caveat on TOCTOU and local trust is appropriate. This review does not claim historical execution is cryptographically proven.
