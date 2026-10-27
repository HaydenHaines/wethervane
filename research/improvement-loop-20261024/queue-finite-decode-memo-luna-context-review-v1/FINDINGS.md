# Complementary ContextVar review of queue decode memo

I checked the frozen queue memo source pins and used a copied source tree for three small invented context-lifetime probes. All 15 entries in the packet's `source-pins.json` match. The probe imports only the copied `execution.py`, `ledger.py`, and `run_receipts.py` bytes after verifying hashes. No queue job, owner/model code, or real store was run.

## Findings

- A fresh ordinary `threading.Thread` starts with no decode memo in this Python runtime. Two decodes in that thread do not increment the parent scope's cache-hit count.
- A deliberate `contextvars.copy_context()` copies the reference to the same mutable memo object. The copied context can use it after the owning context has exited and called `memo.clear()`. In the probe, the copied context repopulates the memo while its `cleared` flag remains true; a subsequent `decode_operation()` in that stale context is rejected as nested.
- An `asyncio.create_task()` created inside the scope inherits the same memo reference. If it suspends until after the owning synchronous scope exits, it resumes with the cleared memo, can repopulate it, and gets the same nested-scope rejection on its next operation.

These confirm the documented boundary, not a current queue-path defect. `status`, `tick`, and `finalize` are synchronous and contain no `await`; their managed scope does not yield to an asyncio task. Fresh ordinary threads are isolated. The README expressly excludes arbitrary copied-context/async service use. However, future async wrappers, manually copied contexts, or concurrent copied thread contexts must not use this API as-is: the memo has no active-generation check or synchronization, and copied contexts can retain/repopulate the mutable object past the owner's cleanup. Add explicit invalidation/context-generation and locking before expanding that boundary.

The primary reviewer is separately testing deepcopy aliasing, PID/fork, byte/item bounds, stale byte/error equivalence, and the full test/regression suites; this packet makes no independent claim on those areas. The review remains within the source's stated synchronous scope.
