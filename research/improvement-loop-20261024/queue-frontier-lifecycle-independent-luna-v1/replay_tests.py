"""Replay frozen lifecycle tiny tests from a fresh disposable copy."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

BASE = Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024')

def main():
    with tempfile.TemporaryDirectory(prefix='luna-frontier-review-') as td:
        lab = Path(td) / 'lab'
        q, a = lab / 'queue-v2', lab / 'queue-incremental-index-sol-v2'
        d, old = lab / 'queue-frontier-lifecycle-sol-v1', lab / 'queue-durable-frontier-sol-v1'
        for path in (q, a, d, old):
            path.mkdir(parents=True)
        for name in ('execution.py', 'experiment_queue.py', 'ledger.py', 'run_receipts.py'):
            shutil.copy2(BASE / 'queue-v2' / name, q / name)
        shutil.copy2(BASE / 'queue-incremental-index-sol-v2/core.py', a / 'core.py')
        shutil.copy2(BASE / 'test_experiment_queue.py', lab / 'test_experiment_queue.py')
        for name in ('dispatch.py', 'test_lifecycle.py'):
            shutil.copy2(BASE / 'queue-frontier-lifecycle-sol-v1' / name, d / name)
        shutil.copy2(BASE / 'queue-durable-frontier-sol-v1/dispatch.py', old / 'dispatch.py')
        env = os.environ.copy()
        for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
            env[key] = '1'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        proc = subprocess.run(
            ['timeout', '--kill-after=3s', '60s', 'nice', '-n', '15',
             'python3', '-B', 'test_lifecycle.py'],
            cwd=d, env=env, text=True, capture_output=True)
        print(proc.stdout, end='')
        print(proc.stderr, end='')
        print('EXIT', proc.returncode)
        return proc.returncode

if __name__ == '__main__':
    raise SystemExit(main())
