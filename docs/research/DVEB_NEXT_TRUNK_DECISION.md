# DVEB / PyStatistics next-trunk decision

**Date:** 2026-09-02

**Status:** next suitability question selected; implementation and timing not
yet authorized

**Research branch:** `research/dveb-pystatistics`

## Outcome

The direct MVN-MLE line is complete as a research milestone. The next bounded
question remains **batched exact ARMA likelihood evaluation**, the sole
unresolved candidate admitted by the original Phase-0 inventory.

This is a selection of what to test next, not a claim that DVEB should replace
the current time-series backend. It does not authorize an exact-batch public
API, a DVEB compiler change, a PyStatistics backend, or performance timing.

## Why MVN-MLE changes the evidence without changing the rules

The original Phase-0 report did not recommend direct MVN-MLE as the first
compiler trunk because dense Cholesky and automatic differentiation are mature
PyTorch capabilities. The later, separately frozen work found a narrower route:
DVEB emitted the complete checked CPU value-and-gradient body, reused the
existing SciPy optimizer, shipped in a small torch-free wheel, and beat the
optimized CPU-PyTorch path across the admitted grid.

That result does not license a blanket replacement program. It establishes a
useful selection rule for the next trunk:

> Look for a complete numerical body whose scientific control flow can be
> expressed directly, whose ownership and scheduling can be compiled once,
> and whose native artifact provides either measured performance or deployment
> value without reproducing a mature library wholesale.

Batched exact ARMA satisfies that rule provisionally. Sparse direct linear
algebra does not yet: a competitive sparse factorization is primarily a
specialist-library problem and should enter DVEB through a future general
library/FFI strategy, not a new CHOLMOD-class implementation in the compiler.

## Why exact batched ARMA is the next question

PyStatistics already has both halves of the scientific justification:

- an exact single-series likelihood with a pure NumPy reference, bit-controlled
  Cython implementation, reusable workspace, and R 4.5.2 fixtures; and
- a public many-series `arima_batch` interface whose exact `method="ml"` path
  is explicitly absent while its Whittle approximation demonstrates a real
  batch axis.

The execution shape is also distinct from the completed dense program. One
series is a sequential Kalman recurrence. Many series are mutually independent,
so a compiler can map complete recurrences across CPU workers or CUDA workers
without changing the statistical algorithm or parallelizing time within a
series.

The implicated language features are general scientific-computing features:

- caller-owned float64 rank-2 arrays;
- a parallel map over independent records;
- bounded local vectors and matrices with indexed access;
- data-dependent control flow inside one worker;
- mutable worker-local scratch with explicit ownership;
- local reductions and finite checks;
- per-record status output; and
- deterministic CPU/CUDA scheduling with legal overrides.

No compiler rule may mention ARMA, Kalman, PyStatistics, or a consumer symbol.
The statistical formulation belongs in a DVEB example/consumer program; only
the general mechanisms belong in the language and compiler. Existing GradFlow
and MVN-MLE regression contracts must remain green.

## Required order of work

1. **Consumer protocol.** Review and approve a prospective Phase-0B protocol
   on this branch. Freeze inputs, formulations, comparators, correctness gates,
   calibration/evaluation separation, and a binary decision rule before code
   or timing.
2. **Workload/null preflight.** Establish the strongest competent existing
   paths: NumPy authority, Cython incumbent, batched multicore native control,
   and ordinary PyTorch eager/`torch.compile` in float64. A PyTorch compiler
   success is a valid Case-1 result.
3. **Native headroom diagnostic.** Only if separately authorized, write a
   formulation-matched CPU/CUDA diagnostic from the declared equations. It is
   a ceiling, not DVEB and not a production backend.
4. **DVEB capability decision.** If the workload has native headroom and DVEB
   cannot express it, record Case 2 and authorize only the smallest coherent
   general capability trunk in `/mnt/projects/dveb`.
5. **DVEB consumer qualification.** Return the versioned handoff to this branch,
   integrate behind an explicit research-only selector, run correctness before
   timing, and apply the frozen rule without post-result optimization.
6. **End-to-end fitting is later.** A fixed-parameter likelihood GO may justify
   a separate fit-level protocol; it does not automatically justify or ship an
   optimizer/backend.

## Decision standard

The governing standard is DVEB's established **non-domination** requirement,
not a demand that every lane beat PyTorch:

- an order-of-magnitude loss in any admitted primary lane is catastrophic;
- systematic material loss over the primary class is NO-GO;
- clear speed wins are preferred and reported, but are not mandatory where the
  same general artifact establishes a meaningful deployment or cross-device
  benefit;
- matching a native ceiling is not required, but unexplained distance from it
  is evidence about emitted code or scheduling that must be attributed before
  blaming the language; and
- correctness precedes performance without exception.

For this candidate, merely matching the existing Cython single-series loop at
`K=1` proves little. The primary question is whether the independent-series
axis produces a useful many-series operating region. A result that accelerates
only a synthetic fixed-parameter kernel, with no credible route to fitting,
is insufficient.

## Guardrails inherited from the completed milestone

- PyPI 6.1.5 and `main` remain frozen.
- Work occurs only on `research/dveb-pystatistics` and a separately authorized
  `codex/` DVEB branch.
- No merge, release, or public-package upload is authorized.
- PyTorch remains the incumbent and must receive its strongest ordinary route.
- No custom Triton/CUDA code may be called “ordinary PyTorch.”
- CPU-resident, CUDA-resident, transfer-inclusive, artifact-preparation, and
  fresh-process costs are separate endpoints.
- Calibration and decision points are disjoint.
- Failed candidates are not timed.
- Negative evidence is preserved with the same detail as a positive result.

## Immediate next deliverable

The next deliverable is a full Phase-0B protocol that turns
`docs/research/DVEB_PHASE0.md`'s outline into a frozen experiment: exact grid,
parameter families, input provenance, comparator identities, tolerances,
repeat structure, thermal rules, deployment endpoints, and numeric Case-1 /
Case-2 / GO / NO-GO thresholds.

That protocol must be reviewed before any comparator, diagnostic ceiling,
DVEB capability, or timing campaign is implemented.

