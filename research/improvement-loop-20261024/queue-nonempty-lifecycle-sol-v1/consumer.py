"""Direct machine-consumer fixture contract; no service shutdown or science approval."""
def accepted_unqualified_work(snapshot):
    if not isinstance(snapshot,dict)or snapshot.get('schema')!='NonemptyCurrentAdmissionsLifecycleV1':raise ValueError('explicit nonempty lifecycle schema required')
    if {'all_terminal','all_executions_accepted_unqualified'}&set(snapshot):raise ValueError('legacy ambiguous completion fields rejected')
    fields=['total','pending','active','executed_terminal','blocked_terminal','accepted_execution','failed_execution']
    if any(type(snapshot.get(k))is not int or snapshot[k]<0 for k in fields):raise ValueError('integer lifecycle counts required')
    total,pending,active,released,blocked,accepted,failed=[snapshot[k]for k in fields]
    if active>1 or total!=pending+active+released+blocked or released!=accepted+failed:raise ValueError('lifecycle count invariant')
    terminal=total>0 and pending==0 and active==0;success=terminal and failed==0 and blocked==0
    if type(snapshot.get('nonempty_current_admissions_all_terminal'))is not bool or snapshot['nonempty_current_admissions_all_terminal']!=terminal:raise ValueError('nonempty terminal flag mismatch')
    if type(snapshot.get('nonempty_current_admissions_all_accepted_unqualified'))is not bool or snapshot['nonempty_current_admissions_all_accepted_unqualified']!=success:raise ValueError('accepted-unqualified flag mismatch')
    if total==0:expected={'empty_no_admitted_work'}
    elif terminal:expected={'nonempty_current_admissions_terminal_all_accepted_unqualified'if success else'nonempty_current_admissions_terminal_with_failures_or_blocks'}
    elif active:expected={'nonempty_current_admissions_active'}
    else:expected={'nonempty_current_admissions_runnable','nonempty_current_admissions_waiting_unresolved'}
    if snapshot.get('state')not in expected:raise ValueError('explicit lifecycle state mismatch')
    return success
