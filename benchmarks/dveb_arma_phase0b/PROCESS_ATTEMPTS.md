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
