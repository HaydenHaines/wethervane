"""Tiny ContextVar isolation probe for the frozen queue decode memo."""
from __future__ import annotations
import asyncio
import contextvars
import hashlib
import json
import sys
import threading
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/queue-finite-decode-memo-sol-v1')
EXPECTED={
    'execution.py':'6cb2a3e94be5326e94447fe25df40333498d405978b20705500c96a64ac233f0',
    'run_receipts.py':'a7e78b822207109e0257adce9abb6493c53a0d93158177131ade76c0d874e73e',
    'ledger.py':'d65cea890ecf9b456f8b9625d173f15dac1e7fa94bb618d3407e26f368ee220a',
}
for name,digest in EXPECTED.items():
    raw=(SOURCE/name).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError(f'frozen source pin mismatch: {name}')
sys.path.insert(0,str(HERE/'scratch'))
import execution as ex

DATA=b'{"tiny": [1]}'
result={}

# A normal independent thread has an empty ContextVar and parses without memo.
with ex.decode_operation() as owner:
    ex.decode(DATA)
    observed={}
    def independent_thread():
        observed['memo_absent']=ex._DECODE_MEMO.get() is None
        ex.decode(DATA); ex.decode(DATA)
    thread=threading.Thread(target=independent_thread)
    thread.start();thread.join(timeout=3)
    if thread.is_alive(): raise RuntimeError('thread join timed out')
    result['new_thread_context_is_empty']=observed['memo_absent']
    result['owner_scope_hit_count_after_thread']=owner.stats['hits']
result['owner_scope_cleared']=owner.stats['cleared'] and not owner.values

# Explicit Context copying shares the memo object; it can be used after the
# owner context resets. That use is outside the synchronous API contract.
with ex.decode_operation() as owner:
    ex.decode(DATA)
    copied=contextvars.copy_context()
    copied.run(ex.decode,DATA)
    result['copied_context_shares_object']=copied.run(lambda: ex._DECODE_MEMO.get() is owner)
result['copied_context_retains_cleared_object']=copied.run(lambda: ex._DECODE_MEMO.get() is owner and owner.stats['cleared'])
copied.run(ex.decode,DATA)
result['copied_context_repopulates_after_owner_exit']=bool(owner.values) and owner.stats['cleared']
def enter_scope():
    with ex.decode_operation():
        pass
try:
    copied.run(enter_scope)
except ex.ExecutionError:
    result['copied_context_new_scope_rejected_as_nested']=True
else:
    result['copied_context_new_scope_rejected_as_nested']=False
copied.run(lambda: ex._DECODE_MEMO.set(None))

async def async_context_case():
    ready=asyncio.Event(); resume=asyncio.Event(); observed={}
    async def child():
        observed['task_inherits_same_object']=ex._DECODE_MEMO.get() is memo
        ready.set()
        await resume.wait()
        observed['task_sees_owner_cleared_flag']=memo.stats['cleared']
        ex.decode(DATA)
        observed['task_repopulates_after_owner_exit']=bool(memo.values)
        try:
            with ex.decode_operation(): pass
        except ex.ExecutionError:
            observed['next_scope_refused_in_stale_task_context']=True
        else:
            observed['next_scope_refused_in_stale_task_context']=False
    with ex.decode_operation() as memo:
        ex.decode(DATA)
        task=asyncio.create_task(child())
        await asyncio.wait_for(ready.wait(),2)
    observed['owning_scope_cleared_first']=memo.stats['cleared'] and not memo.values
    resume.set()
    await asyncio.wait_for(task,2)
    observed['outer_context_is_reset']=ex._DECODE_MEMO.get() is None
    return observed
result['asyncio_task_context']=asyncio.run(async_context_case())
print(json.dumps(result,indent=2,sort_keys=True))
