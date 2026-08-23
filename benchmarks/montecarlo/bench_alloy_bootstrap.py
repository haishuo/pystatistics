#!/usr/bin/env python3
"""Gate 6: the same public `boot(...)` call, ALLOY/Metal against PyTorch/MPS.

THE WHOLE OPERATION on both sides -- `boot(data, mean_stat, ..., backend='gpu',
gpu_statistic='mean')`, including `t0`, the declaration check, the resampling
and the summary statistics. Timing only the kernels would measure something no
caller can ask for.

The comparison is made by forcing the backend, since the dispatcher picks by
device and there is only one device here. Both paths run the identical public
entry point otherwise.

ABBA ORDER. Timing every ALLOY run before every PyTorch run lets drift over the
run -- clock ramping, thermal state, another process arriving -- land entirely
on one side. Alternating cancels a monotone drift to first order.

    python benchmarks/montecarlo/bench_alloy_bootstrap.py
"""

from __future__ import annotations

import argparse
import pathlib
import platform
import statistics as st
import sys
import time

import numpy as np

# Run from a checkout without installing: the repository root is two levels up.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from pystatistics.montecarlo import solvers
from pystatistics.montecarlo.backends.gpu import GPUBootstrapBackend
from pystatistics.montecarlo.backends.alloy import ALLOYBootstrapBackend
from pystatistics.montecarlo import boot

CASES = ((100, 10_000), (1_000, 10_000), (10_000, 2_000), (100, 100_000))
SEED = 20260823


def mean_stat(data, indices):
    return np.array([np.mean(data[indices])])


def _run_with(backend_cls, data, n_resamples):
    """One complete public bootstrap, with the GPU backend forced."""
    original = solvers._boot_gpu_backend
    solvers._boot_gpu_backend = lambda: backend_cls()
    try:
        return boot(data, mean_stat, n_resamples=n_resamples, seed=SEED,
                    backend="gpu", gpu_statistic="mean")
    finally:
        solvers._boot_gpu_backend = original


def timed_pair(a_fn, b_fn, warm: int, rounds: int):
    for _ in range(warm):
        a_fn()
        b_fn()
    ta, tb = [], []

    def once(fn, into):
        t0 = time.perf_counter()
        fn()
        into.append((time.perf_counter() - t0) * 1e3)

    for _ in range(rounds):
        once(a_fn, ta)      # A B
        once(b_fn, tb)
        once(b_fn, tb)      # B A
        once(a_fn, ta)
    return ((st.median(ta), min(ta), max(ta)),
            (st.median(tb), min(tb), max(tb)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    from pystatistics.montecarlo.backends import _alloy
    ok, why = _alloy.is_available()
    if not ok:
        print(f"ALLOY backend unavailable: {why}", file=sys.stderr)
        return 77
    try:
        import torch
        tver = torch.__version__
        if not torch.backends.mps.is_available():
            print("MPS unavailable", file=sys.stderr)
            return 77
    except Exception as exc:  # noqa: BLE001
        print(f"PyTorch unavailable: {exc}", file=sys.stderr)
        return 77

    rec = _alloy.provenance()
    print(f"python {platform.python_version()} ({sys.executable})")
    print(f"numpy {np.__version__}   torch {tver}")
    print(f"macOS {platform.mac_ver()[0]} {platform.machine()}")
    print(f"ALLOY commit {rec['alloy_commit'][:12]}   "
          f"compiler {rec['compiler_version']}   "
          f"bundle {rec['bundle_format_version']}")
    print(f"seed {SEED}   warm {args.warm}, {args.rounds} ABBA rounds "
          f"(= {2 * args.rounds} timed runs each)\n")

    print(f"  {'n':>6} {'R':>7} | {'ALLOY/Metal':>22} {'PyTorch/MPS':>22} | "
          f"{'ratio':>6}")
    slowest = None
    for n, R in CASES:
        data = np.random.default_rng(11).normal(10.0, 2.0, n)

        # Correctness before timing: both must be the same statistic.
        ra = _run_with(ALLOYBootstrapBackend, data, 400)
        rt = _run_with(GPUBootstrapBackend, data, 400)
        assert ra.t0[0] == rt.t0[0], "the two backends disagree on t0"
        se = float(np.std(data, ddof=1)) / np.sqrt(n)
        for label, r in (("alloy", ra), ("torch", rt)):
            ratio = float(r.standard_errors[0]) / se
            assert 0.85 < ratio < 1.15, f"{label} SE/analytic = {ratio}"

        (am, alo, ahi), (tm, tlo, thi) = timed_pair(
            lambda: _run_with(ALLOYBootstrapBackend, data, R),
            lambda: _run_with(GPUBootstrapBackend, data, R),
            args.warm, args.rounds)
        ratio = tm / am
        slowest = ratio if slowest is None else min(slowest, ratio)
        print(f"  {n:>6} {R:>7} | {am:>8.3f} [{alo:6.3f}..{ahi:6.3f}] "
              f"{tm:>8.3f} [{tlo:6.3f}..{thi:6.3f}] | {ratio:>5.2f}x")

    print(f"\n  medians in ms, [min..max]. Worst case {slowest:.2f}x.")
    print("  Observation of the complete public call, not a kernel timing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
