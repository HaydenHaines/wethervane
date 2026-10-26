# Tiny no-C gradient runtime probe

The already-installed PyTensor NUMBA linker computes the 40-contribution fixture's analytic gradient correctly. At x=0.37, the sum of squared deviations from offsets 0,0.1,…,3.9 is 153.156 and its derivative is −126.4. NUMBA matches both direct NumPy identities in 2.085 seconds. The same symbolic expression under no-C FAST_COMPILE fails on the 40-input derivative Add with `Scratchpad has no attribute ufunc` in 0.141 seconds; its complete traceback is preserved.

This isolates the previously observed Python-linker large-Add limitation and demonstrates one already-available CPU linker can evaluate the same tiny analytic derivative. It does **not** establish that NUMBA supports every operation, exact semantics, resource demand or gradients of the actual 662-dimensional count graph. No actual graph was imported, rebuilt or rerun.

The existing calibration environment supplied NumPy 2.5.4, PyTensor 2.38.2, Numba 0.68.0 and llvmlite 0.50.0. `result.json` records exact import paths, executable and thread environment. C compilation was disabled (`cxx=`); NUMBA uses its existing LLVM JIT, with caches confined to this directory. No package/environment/owner file changed. The process ran at nice15 with all five configured thread limits set to one.

Reproduce from the lab root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
NUMBA_CACHE_DIR=/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/m01-gradient-runtime-probe-sol-v1/numba-cache \
PYTENSOR_FLAGS='cxx=,floatX=float64,compiledir=/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/m01-gradient-runtime-probe-sol-v1/pytensor-cache' \
nice -n 15 .improvement-ledger/runtime/calibration-v1-env/bin/python -B \
research/improvement-loop-20261024/m01-gradient-runtime-probe-sol-v1/probe.py
```
