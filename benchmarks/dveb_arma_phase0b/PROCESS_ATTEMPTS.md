# Phase-0B preflight process attempts

This file preserves pre-decision implementation and harness failures. None of
these attempts is performance evidence.

## 2026-09-02 — static full-graph Python loop

`torch.compile(fullgraph=True)` of the literal Python time loop did not finish
compiling the smallest C01 shape (`K=8`, `n=120`, `r=1`) within three minutes
and was interrupted. The eager formulation had already passed correctness.
No evaluation result or timing decision existed. The failure is retained as an
operational/compilation finding; the predeclared public `torch.while_loop`
variant remained available for testing.

## 2026-09-02 — first structured-loop indexing form

The first `torch.while_loop` formulation used `z[:, t]`, where `t` was a
carried scalar tensor. PyTorch 2.13 refused it with
`GuardOnDataDependentSymNode`; no fallback occurred. Replacing the Python-style
index with the equivalent public tensor operation
`torch.index_select(z, 1, t.reshape(1)).squeeze(1)` passed C01 and C02 at the
same numerical differences as eager PyTorch. This is a formulation correction,
not a threshold or workload change.

## 2026-09-02 — one-row broadcast ownership

The first full-grid run passed C01--C04, then stopped before evaluating E01.
For `K=1`, `np.ascontiguousarray(np.broadcast_to(...))` retained a read-only
buffer because the broadcast view was already C-contiguous. The Cython memory
view correctly refused it. The parameter bytes and frozen hashes were
unchanged; the generator was corrected to force owning, writable C-contiguous
copies for every K, and the input manifest was refreshed before resuming.

## 2026-09-02 — focused regression runner entry point

The first post-preflight regression rerun invoked the environment's standalone
`pytest` wrapper. It collected zero tests and produced four import errors
because the research worktree was absent from that wrapper's import path. No
test or numerical code executed. Re-running the identical four-file selection
through the environment's `python -m pytest` entry point collected and passed
all 153 tests. The permanent regression record uses the successful,
worktree-aware invocation.

## 2026-09-02 — native L3 campaign cleanup adapter

The first frozen CUDA L3 decision attempt completed its first cell in memory
and then failed during cleanup because `TorchTrace` did not implement the
no-op `close()` method shared by the native adapters. The process wrote no
result file, emitted no cell timings, and exposed no comparative result. The
entire attempt is invalidated. The only harness change adds the missing no-op
lifecycle method; no numerical operation, candidate, workload, threshold,
order seed, warmup, or observation rule changed. A v2 source freeze records
one invalidated attempt and zero retained decision observations before rerun.

## 2026-09-02 — L3 analysis JSON scalar conversion

The completed v2 timing evidence remained intact, but the first mechanical
analysis attempt stopped before writing a file or printing a verdict because
NumPy boolean scalars are not JSON-serializable. The repair adds explicit
Python `bool(...)` conversions around the already computed comparisons. It
does not change a ratio, bootstrap sample, threshold, or decision expression.

## 2026-09-02 — ancillary CUDA L1/L2 repetition projection

After the complete L3 campaign had established Case 2, the first descriptive
CUDA L1/L2 attempt sized repetitions from the fastest native calibration path.
At E01 this selected 3,369 shared repeats. The transfer-inclusive PyTorch path
made the cell take about 15 minutes, more than 10x beyond the planning model,
and projected an unreasonable duration for the remaining non-decision cells.
The attempt was interrupted after E01 completed in memory. It wrote no result
file, printed no endpoint comparison, and retains zero observations.

The authorized duration review caps the fresh ancillary campaign at 64 shared
repeats per cell while preserving 10 warmups, 30 paired randomized
observations, identical work between implementations, all correctness checks,
the fixed grid, and the already completed L3 decision. This is a duration-only
repair; it does not revise or rerun decision evidence.

The 64-repeat correction remained too large: E02 was still running after six
minutes. The attempt was interrupted under the same duration rule. Its
checkpoint contains only E01; it was not inspected and is invalidated in full.
No E02 result completed. A second prospective duration-only correction reduces
the shared cap to 8 repeats. All other controls listed above remain unchanged.

## 2026-09-02 — ancillary launcher package path

The first launch of the final ancillary harness used `PYTHONPATH=src:...`, but
this repository has a root-level `pystatistics/` package rather than a `src/`
layout. Import refused immediately with `ModuleNotFoundError`; no numerical
code, warmup, or timing ran and no output file was created. The unchanged
committed harness was relaunched with the repository root on `PYTHONPATH`.
