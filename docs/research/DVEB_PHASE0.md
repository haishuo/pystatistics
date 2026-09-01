# DVEB / PyStatistics Phase-0 suitability report

**Date:** 2026-09-01

**Status:** Phase-0 inventory complete; no backend implementation authorized

**Branch:** `research/dveb-pystatistics`

**Released base:** `v6.1.5` (`d68f011c287e7af0ba23d874e675ce48cb923226`)

**Research-line identity commit:** `ee65cecb2ac194bac7d3b8424b7496cf79ef6a84`

**Research version:** `6.2.0.dev0+dveb`

## Binary outcome

**CONDITIONAL GO to one separately approved Phase-0B suitability screen:**
batched exact ARMA likelihood evaluation via the existing Kalman formulation.

This is not authorization to implement a PyStatistics backend or new DVEB
compiler capability. It is not evidence that DVEB is faster than PyTorch,
Cython, SciPy, or a native CUDA implementation. It means only that one real
PyStatistics boundary has the right workload shape and enough independent
validation evidence to justify a prospective, deliberately losable screen.

All other candidates examined here are either already well served by
competent PyTorch/NumPy/SciPy paths, too small or too sequential at their real
production sizes, insufficiently grounded in a current product workload, or
principally a numerical-library problem rather than a language/compiler
problem.

## Publication firewall

This research line exists because the forward-Cholesky MVN-MLE paper is pinned
to the package published on PyPI as version 6.1.5. The journal editor requires
the submitted package and the published package to match and also requires a
reproducible, public-data, step-by-step applied example.

The following rules therefore govern this entire research line:

1. `main`, tag `v6.1.5`, and the PyPI 6.1.5 artifact are publication
   infrastructure, not development surfaces.
2. DVEB-related PyStatistics work occurs only on
   `research/dveb-pystatistics`, and only where DVEB intersects the library.
3. The branch must never be merged to `main` during the publication freeze.
   It must not be tagged or released as a public PyStatistics version.
4. Branch-only pushes to `origin/research/dveb-pystatistics` are permitted.
   Such a push does not alter `main`, an existing tag, or PyPI.
5. The distinct version `6.2.0.dev0+dveb` must remain visible so a research
   installation cannot be mistaken for the released package.
6. The ALLOY research line remains separate. Reconciliation, if ever desired,
   requires a new integration branch and an explicit review after the danger
   period; neither research line is merged directly into the other.
7. No research result may require changing the paper's frozen replication
   environment. The released 6.1.5 behavior remains the publication oracle.

These are project-governance rules, not Git limitations. Git can safely retain
and publish the branch while leaving the release line unchanged.

## Scope and evidence

Phase 0 was read-only except for this report. It inspected:

- the complete PyStatistics 6.1.5 source tree and test suite;
- the existing lower-level acceleration audit;
- current benchmark and validation surfaces;
- the four compiled Cython recurrence/risk-set implementations;
- current PyStatistics GPU backends and explicit CPU-only boundaries;
- PyStatsBio only as read-only evidence of downstream workload shapes;
- the current DVEB compiler/runtime capability boundary and the completed
  GradFlow/DVEB scheduling evidence.

No numerical implementation, prototype, benchmark, dependency installation,
or generated DVEB artifact was created in PyStatistics. No PyStatsBio,
GradFlow, ALLOY, DVEB, validation, release, tag, or `main` state was modified.

### Evidence map

The load-bearing local sources are:

- `pystatistics/timeseries/_arima_batch.py` — existing many-series contract,
  Whittle-only boundary, and explicit exact-ML refusal;
- `pystatistics/timeseries/_arima_kalman.py` and
  `_arima_kalman_kernel.pyx` — exact formulation, workspace ownership, and
  compiled recurrence;
- `tests/timeseries/test_arima_kalman_cython_parity.py` — bit-identity oracle;
- `tests/timeseries/test_arima_kalman_r_parity.py` and
  `tests/fixtures/arima_kalman_r_reference.json` — R 4.5.2 evidence;
- `pystatistics/timeseries/backends/whittle_batch_gpu.py` — strongest existing
  PyTorch many-series path and compiled-NLL precedent;
- `pystatistics/mixed/_struct_sparse.py` — the sparse direct-factorization
  contract;
- `docs/ACCELERATION_AUDIT_2026-08.md` — earlier formulation, framework, and
  scale audit that this report does not repeat;
- DVEB `README.md`, `docs/CHARTER.md`, and trunk-005 records — current language,
  ABI, scheduling, and nonclaim boundary.

### Established facts carried forward

- PyTorch is an optional PyStatistics dependency; the base CPU installation
  remains NumPy/SciPy plus the shipped Cython extensions. A lean DVEB path
  could reduce the *GPU extra's* deployment surface, but it would not make the
  base package newly independent of PyTorch.
- The local CUDA PyTorch research environment occupies about 4.8 GiB. That is
  deployment evidence, not a performance result and not by itself a reason to
  build a language backend.
- The August acceleration audit already tested the broad claim that lower-level
  CUDA should replace current PyTorch paths. Dense GPU work is generally
  compute-bound inside PyTorch's optimized linear-algebra kernels, while
  `torch.compile` captured the useful fusion in the Whittle likelihood.
- DVEB's GradFlow work established that it can emit strong CPU/CUDA code,
  approach a separately authored CUDA ceiling, expose native ABIs, and make
  explicit residency decisions. It did not establish statistical data types,
  matrix algebra, automatic differentiation, random-number facilities, or a
  Python extension ABI.
- DVEB's current portable frontend is a bounded three-dimensional float32
  pipeline frontend. The older trunk-001 frontend admits f64 scalar stencil
  expressions, but neither frontend presently expresses the candidate chosen
  below. That gap is Case 2 until a workload screen establishes that the
  required capabilities are worth adding.

## Workload inventory

| Workload boundary | Current competent path | DVEB reading | Phase-0 disposition |
|---|---|---|---|
| Dense LM/GLM, PCA, ordinal and multinomial likelihoods | PyTorch matmul, factorization, reductions, and compiled elementwise graphs | PyTorch home turf; a DVEB win would require competing with mature BLAS/solver kernels | Do not screen |
| Direct forward-Cholesky MVN-MLE | PyTorch f64 objective/autograd on CUDA plus host SciPy optimization | Central to the paper, but dense Cholesky and AD are strong PyTorch capabilities; DVEB has no AD | Do not use as the first DVEB trunk; preserve as publication workload |
| EM MVN-MLE and MICE | Batched PyTorch linear algebra, categorical methods, RNG, missing-pattern machinery | Real and important, but would require several language/runtime capabilities at once; prior profiling says the hot path is already dominated by optimized batched linear algebra | Do not screen first |
| Batched Whittle ARMA | Batched FFT + Adam in PyTorch; `torch.compile` already improves the NLL | Existing successful PyTorch path and an approximate likelihood, not the missing exact path | Comparator and product precedent, not a DVEB target |
| Exact single-series ARIMA and ETS | Bit-controlled Cython scalar recurrences driven by SciPy optimizers | A single dependency chain belongs on a CPU core at ordinary series lengths | Do not GPU-screen as a single series |
| **Batched exact ARMA likelihood** | **No shipped batch path; `arima_batch(..., method="ml")` refuses** | **Sequential within each series, independent across series; one native worker can own one recurrence and its local state** | **Only admitted Phase-0B candidate** |
| Crossed/nested LMM/GLMM | SciPy sparse matrices + SuperLU with fill-reducing ordering and log determinant | Genuine missing PyTorch capability, but the hard part is a sparse direct-solver library | Route to library/FFI investigation, not compiler trunk |
| Single-factor LMM and batched tiny SPD solves | Structured CPU path or PyTorch/cuSOLVER batched factorizations | A possible specialist-kernel niche, but no current evidence prices a new implementation above existing libraries | Park pending a measured product bottleneck |
| Cox/KM/concordance and exact ARIMA/ETS for one dataset | Prefix/risk-set methods or Cython recurrence/Fenwick loops | GPU parallel forms exist in theory, but current production sizes are below the measured crossover | Park with explicit scale trigger |
| Fixed-margin Monte Carlo | Vectorized SciPy Patefield on CPU; ordinary PyTorch Boyett covers the tensor-friendly region | Previous Phase-0A found no production-relevant GPU gap | Permanent NO-GO absent new workload evidence |
| Arbitrary user-defined resampling statistics | CPU Python callable, with declared tensor statistics accelerated separately | A language could compile the statistic, but this first requires a real user program and a broad user-facing surface | No present trunk |

## Why sparse direct linear algebra is not the first trunk

PyStatistics' crossed/nested mixed-model path repeatedly forms and factors

```text
M = Lambda' Z' W Z Lambda + I
```

and needs solves, a fill-reducing ordering, reusable factorization structure,
and a log determinant. Shipped PyTorch does not provide the complete sparse
direct-factorization contract needed here. That is a genuine capability gap.

It is nevertheless primarily a **numerical-library capability**, not evidence
that DVEB should implement a sparse multifrontal solver. A competent solution
would normally call an established implementation such as SuiteSparse on the
CPU or an appropriate supported NVIDIA sparse-direct library on CUDA, with
careful symbolic-factor reuse and ordering control. DVEB may eventually need
an FFI/library-selection layer capable of expressing and scheduling such a
call. Reimplementing CHOLMOD/SuperLU-class machinery inside the compiler would
be an unjustified research project and would violate the rule that DVEB should
not duplicate mature specialist libraries.

The sparse path becomes a DVEB candidate only after a separate investigation
shows that a suitable library exists with the required factorization, logdet,
precision, redistribution, and licensing contract, and that DVEB adds value by
coordinating it with surrounding computation. It is not trunk material now.

## The admitted candidate: batched exact ARMA likelihood

### Existing product boundary

PyStatistics already exposes `arima_batch(Y, ...)` for `K` independent series.
The shipped implementation is Whittle-only and expressly rejects
`method="ml"`. The documentation reports a crossover around `K = 100` and
uses `K = 1000, n = 2000` as a real batched operating point. Therefore the
batch axis is not invented for this screen.

The exact single-series implementation is unusually strong as an oracle:

- a pure NumPy reference and a Cython implementation execute the same scalar
  operation order;
- wheel tests require bit identity with contraction disabled;
- R 4.5.2 fixtures cover non-seasonal and seasonal ARIMA, forecasting,
  likelihood, standard errors, persistent seasonal roots, and failure paths;
- the production Cython workspace fuses stationary initialization, diffuse
  fallback, the Kalman recurrence, and reductions without per-evaluation
  allocation;
- full-fit tests verify that workspace reuse preserves the optimizer
  trajectory and final result.

### Why the shape differs from the rejected single-series GPU idea

For one series, every time step depends on the preceding Kalman state. A GPU
cannot parallelize that dependency chain merely by translating it to CUDA.
For `K` independent series, however, the useful parallel axis is `K`: each CPU
worker or CUDA thread/block can own the complete sequential recurrence for one
series, keep its small state and covariance local, and return one likelihood
and status. No synchronization is needed between series.

This is not an associative scan proposal and does not alter the statistical
algorithm. It preserves the exact time-domain likelihood and changes only how
independent likelihoods are scheduled.

### Why this is not already a DVEB GO

The full statistical fit is more than its recurrence. The present exact path
uses per-series SciPy L-BFGS-B with numerical gradients, and different series
converge at different times and may enter invalid parameter regions. A useful
backend must eventually account for optimizer control flow, failure masks,
Hessian/standard-error work, and result interpretation. Accelerating a frozen
parameter trace while making no credible path to an end-to-end fit would not
justify a PyStatistics backend.

The existing `arima_batch` API also demonstrates demand for many series, but
the repository contains no real-data exact-batch use case. Phase 0 therefore
admits only a suitability screen. Product adoption would require a public,
real-data use case and an interpretation-oriented example independently of
the performance result.

## Minimum general DVEB capabilities implicated

The candidate would require language/compiler features that are independently
useful for scientific computing and must not be named or hardcoded for ARIMA:

1. f64 rank-2 caller-owned arrays with explicit strides or a deliberately
   contiguous contract;
2. a parallel map over independent records/series;
3. data-dependent `for`/`while` control flow within one worker;
4. fixed-size or compile-time-bounded local vectors and matrices with indexed
   access;
5. local mutation and scratch reuse with explicit ownership;
6. reductions local to one worker, including `log` and finite checks;
7. per-record status/failure output without silent fallback;
8. deterministic CPU/CUDA schedule selection and legal forced overrides;
9. a versioned Python-consumable C ABI with caller-owned host and resident
   device entry points;
10. an FP policy capable of preserving the required operation order on the
    CPU reference path and predeclaring any CUDA tolerance before results.

Automatic differentiation, a tensor library, random-number generation, sparse
factorization, and a general optimizer are not prerequisites for the initial
likelihood screen. They may become prerequisites for a complete backend; the
screen must not silently expand to include them.

## Proposed Phase-0B suitability screen — not yet approved

No implementation or timing may begin until a prospective protocol is
reviewed, approved, and committed on this research branch and a corresponding
bounded DVEB trunk is separately authorized in the DVEB repository.

### Scientific question

> Can a direct, maintainable DVEB program execute many independent exact
> float64 ARMA Kalman likelihoods near native CPU/CUDA speed, and does that
> execution shape expose a material limitation of competent ordinary PyTorch
> rather than merely reproduce a fast Cython CPU loop?

### Workload boundary

- Exact Gaussian time-domain ARMA likelihood only: stationary initialization,
  diffuse fallback, forward Kalman recurrence, concentrated variance, log
  determinant contribution, and per-series status.
- Identical input series and parameter vectors for every implementation.
- No Whittle substitution, no fp32 downgrade, no altered initialization, no
  changed convergence test, and no hidden CPU fallback.
- The first screen evaluates likelihood batches at fixed parameter proposals.
  A GO schedules, but does not itself authorize, an end-to-end batched-fit
  screen.

### Required comparators

1. Pure NumPy reference: correctness authority and readable formulation.
2. Existing Cython workspace: competent native CPU incumbent.
3. A competent batched CPU control that exploits independent series rather
   than calling the Python API serially.
4. Ordinary PyTorch eager and `torch.compile(fullgraph=True)` using public
   tensor operations, f64 CUDA, no custom CUDA/Triton/operator code.
5. A separately authored, formulation-matched CUDA diagnostic ceiling, only
   if separately authorized; it is not the primary comparator.

The PyTorch implementation must be written from the declared Kalman equations
and checked for graph breaks, recompilation, kernel count, transfers, and
materialized state. If TorchInductor produces an efficient per-series kernel,
that is a valid Case-1 result, not cheating.

### Candidate grid

The final grid and calibration/evaluation split must be approved before code.
A reasonable starting domain, grounded in shipped tests and documentation, is:

- batch count `K` in `{1, 16, 64, 256, 1024}`;
- series length `n` in `{120, 500, 2000, 10000}`;
- state dimension `r` in `{1, 3, 13, 25}`;
- stable, persistent-near-boundary, and invalid/diffuse-fallback parameter
  families;
- disjoint calibration and evaluation cells, with no post-result grid edits.

`K = 1` characterizes the single-series CPU region and cannot establish a GPU
win. Primary decision cells must have a production-credible many-series batch
and must include `K = 1000, n = 2000` or the nearest mechanically admitted
cell because that shape already exists in the shipped batch-ARIMA evidence.

### Correctness gates before timing

- exact equality with the pure NumPy reference on the contraction-disabled
  CPU path wherever the same operation order is emitted;
- exact Cython agreement for stationary initialization, recurrence outputs,
  local scratch mutation, status, SSE, and sum-log-F at the existing parity
  fixtures;
- predeclared f64 CPU/CUDA tolerances derived on disjoint qualification data,
  never loosened after evaluation;
- R-fixture agreement for likelihood-bearing full fits run through the
  existing reference path;
- explicit persistent-root, nonstationary, diffuse-fallback, non-finite,
  wrong-shape, stride, and local-capacity refusal tests;
- identical auto and forced-policy mathematics;
- conservation is not applicable and must not be invented as a gate.

A candidate that fails correctness is not timed.

### Performance endpoints

Report separately:

- resident repeated likelihood evaluation;
- host-to-device plus resident evaluation plus result materialization;
- artifact/library preparation and size;
- fresh-process import/load-to-answer;
- CPU one-thread and representative multicore execution;
- peak memory, local-memory spills, registers, launch count, and selected
  automatic schedule.

Steady-state kernel speed, end-to-end statistical usefulness, and deployment
footprint are different claims. None may substitute for another.

### Prospective decision shape

Exact numeric thresholds remain **unapproved**. The eventual binary rule must
require all of the following for a GO:

1. DVEB materially outperforms the best competent ordinary PyTorch path over
   a predeclared majority of primary many-series evaluation cells, without a
   material-regression cell hidden by aggregation;
2. DVEB is acceptably near the independently authored native CUDA ceiling if
   that ceiling is authorized;
3. DVEB's multicore CPU path is competitive with a competent batched Cython/C
   control in the CPU-admitted region;
4. memory, transfers, launch count, and compilation/load costs do not erase
   the resident result for the intended use pattern;
5. the evidence supports a credible next step toward end-to-end exact batch
   fitting rather than only a synthetic fixed-parameter microkernel.

**Case 1 / NO-GO:** ordinary PyTorch compiles the recurrence efficiently, the
native CPU incumbent remains preferable at real sizes, or the end-to-end path
has no useful operating region. Preserve the result and do not build the
backend.

**Case 2:** the workload has independently demonstrated native headroom, but
one or more of the bounded general DVEB capabilities above is missing. Schedule
only those capabilities; do not disguise capability absence as workload
failure.

## Phase-0 interpretation

PyStatistics should not become a blanket argument for replacing PyTorch. The
evidence says almost the opposite: PyTorch has absorbed most dense,
tensor-expressible statistical workloads remarkably well. DVEB earns a place
only where the scientific program has useful native structure that a tensor
framework cannot recover as effectively, or where a native artifact provides
a separately measured operational benefit.

Batched independent recurrences are the one current boundary that warrants
testing that proposition. Sparse direct linear algebra warrants a library
strategy. Everything else stays with its competent incumbent until real
workload evidence changes the decision.

The correct Phase-0 conclusion is therefore neither "DVEB replaces PyTorch"
nor "DVEB has no role": it is **one bounded, falsifiable next question**.
