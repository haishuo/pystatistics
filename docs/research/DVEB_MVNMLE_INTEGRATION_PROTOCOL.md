# DVEB forward-Cholesky MVN-MLE integration protocol

**Frozen:** 2026-09-01, before consumer-adapter implementation or timing

**Status:** PROSPECTIVE RESEARCH-BRANCH PROTOCOL

**Branch:** `research/dveb-pystatistics`

**Released base:** PyStatistics `v6.1.5`

## Question and decision

Can the qualified DVEB Trunk 008 CPU-only dense artifact serve as an explicit,
optional implementation of PyStatistics's existing forward-Cholesky direct
MVN-MLE objective while preserving the released statistical contract, working
without PyTorch, and avoiding material end-to-end domination by the current
competent CPU-PyTorch path?

The result is **GO** only if all artifact, API, numerical, optimizer,
deployment, regression, and non-domination gates below pass. A GO admits an
experimental branch-only `solver='dveb'` route. It does not change the default,
replace PyTorch, modify the paper package, authorize a release, or authorize a
merge to `main`.

The result is **NO-GO / Case 1** if the adapter is correct but the frozen
artifact is materially dominated end to end. It is **HALT / Case 2** if the
handoff lacks a required general ABI capability or the prospective protocol is
invalid before timing. No threshold or evaluation point may be changed after
comparative results exist.

## Publication firewall

This work occurs only on `research/dveb-pystatistics`, whose visible version is
`6.2.0.dev0+dveb`. Tag `v6.1.5`, PyPI, `main`, the paper replication
environment, the ALLOY research line, GradFlow, PyStatsBio, and the DVEB
repository are read-only inputs. The branch must not be merged, tagged, or
released during the publication freeze. A later branch push is permitted only
when explicitly authorized.

## Immutable handoff

The consumer authority is DVEB commit `a4b0286`, with Trunk 008's qualified
handoff at `evidence/trunk008/handoff/manifest.json`:

- bundle SHA-256:
  `61faedbed04e0e882109b25fe3139d0315c6c16e95f6356eefcd74a51fdadfa5`;
- native library SHA-256:
  `a96e410282fae692f532185ba9c6d0377885f9f7f2b696ff772eda13b1fb102f`;
- ABI version: 1;
- language-source SHA-256:
  `a6ee8e60f6ed52e991ff53b12b1bee5ef6781a0f9506b324db18740e9227dc91`;
- implementation commit:
  `dcfd892564177343cb918efbafaefca2f28291de`;
- target and precision: Linux x86-64 CPU, IEEE float64,
  `-ffp-contract=off`, no fast-math;
- dependencies: ordinary C/C++/math/OpenMP runtime only; no PyTorch, CUDA, or
  NVIDIA dependency.

The exact known-machine library and a consumer manifest may be copied into
this research branch. Every copied member is hash-verified before first use.
The adapter refuses a wrong ABI, hash, platform, dtype, layout, shape, alias,
schedule, or non-finite input. It never searches arbitrary system library
paths and never falls back silently.

The known-machine artifact is not a portable Linux promise. This research
integration may qualify it on Forge only. A release would require separately
qualified per-platform artifacts, correct wheel tags, and a new packaging
protocol.

## Ownership boundary

PyStatistics owns:

- `MVNDesign`, missingness-pattern extraction, and sufficient-statistic
  preparation;
- public argument validation and degeneracy checks;
- initial parameters, SciPy optimizer policy, convergence criteria, result
  interpretation, diagnostics, and `MVNSolution`;
- explicit selection of the experimental route;
- loading and error mapping for the native artifact.

DVEB owns only the checked value-and-gradient call and its context scratch.
It does not know PyStatistics APIs, select an optimizer, interpret a fit, or
query hardware inside evaluation.

## Research API

The only user-visible selection added by this branch is
`mlest(..., method='direct', backend='cpu', solver='dveb')`; omitting
`backend` is also legal because CPU is the default. `solver='dveb'` with
`method='em'` or `'monotone'`, or with `backend='gpu'`, `'gpu_fp64'`, or
`'auto'`, is rejected. No default or automatic route selects DVEB.

The adapter exposes automatic, forced-serial, and forced-work-item-parallel
schedules internally for qualification. Public `mlest` uses automatic
scheduling and the process CPU affinity determined once at objective setup.
There is no profiling, compilation, environment query, or candidate racing in
the numerical call.

## Functional qualification before timing

### F1 — artifact and loader

- verify the committed consumer manifest and native library hashes;
- verify ABI version and all required symbols;
- verify Linux/x86-64 eligibility before loading;
- verify no CUDA, NVIDIA, PyTorch, RPATH, or RUNPATH dependency;
- map every DVEB status to a truthful PyStatistics exception;
- prove context destruction is idempotent at the Python ownership boundary.

### F2 — objective and gradient

On disjoint qualification cases Q1--Q4 from the DVEB handoff, and on the
released apple and missvals fixtures where applicable:

- objective error at most `1e-9 + 1e-9 * abs(F_ref)`;
- gradient max-absolute error at most
  `max(1e-9, 2e-8 * max(abs(g_ref)))`;
- finite-difference gradient checks on Q1, Q3, and Q4;
- automatic and both forced schedules return admitted mathematics;
- repeated calls are stable for a fixed schedule and thread count;
- thread counts 1, 6, and 12 are exercised;
- non-finite, wrong-size, noncontiguous, wrong-dtype, closed-context,
  factorization, and illegal-schedule paths fail loudly.

No tolerance may be relaxed after evaluation results exist.

### F3 — optimizer and public result

- apple direct fitting agrees with the released R fixture and the incumbent
  forward-Cholesky path within existing PyStatistics tolerances;
- the existing direct missvals, large-clean-data, non-convergence,
  degeneracy, and result-interface tests pass through the DVEB route where the
  mathematical contract applies;
- objective scaling, SciPy method, tolerance, maximum iterations, convergence
  flag, warnings, timings, backend name, parameterization label, `muhat`,
  `sigmahat`, and `loglik` remain truthful;
- the optimizer uses the fused value-and-gradient ABI rather than evaluating
  the numerical body twice.

### F4 — torch-free operation

In a process where importing `torch` raises, `solver='dveb'` must import,
construct, fit apple, and return the admitted result. The adapter and artifact
must contain no import, link, string, or fallback dependency on PyTorch. The
existing default behavior remains unchanged and may still warn and use the
NumPy reference when PyTorch is absent.

### F5 — regression firewall

- all MVN-MLE tests pass;
- the complete default non-slow PyStatistics suite passes;
- package version remains `6.2.0.dev0+dveb`;
- `main`, tag `v6.1.5`, and the released behavior are unchanged;
- no file outside the approved PyStatistics research worktree changes.

No performance lane is timed if an applicable functional gate fails.

## Prospective performance campaign

### Implementations

1. incumbent `backend='cpu'` forward-Cholesky FP64 PyTorch;
2. explicit `solver='dveb'` using the frozen Trunk 008 artifact.

Both use the same `MVNDesign`, initial parameter vector, SciPy BFGS driver,
per-observation scaling, tolerance, maximum iterations, and result assembly.
The comparison is an implementation comparison, not a new estimator.

### Evaluation grid

Use the frozen Trunk 008 evaluation generators without changing their seeds:

| Case | n | p | Missingness |
|---|---:|---:|---|
| E1 | 150 | 4 | MCAR 15%, seed 0 |
| E2 | 178 | 13 | MCAR 15%, seed 1 |
| E3 | 569 | 30 | MCAR 15%, seed 2 |
| E4 | 5000 | 13 | structured, at most 32 patterns, seed 3 |
| E5 | 5000 | 30 | structured, at most 64 patterns, seed 4 |
| E6 | 5000 | 30 | MCAR 15%, seed 5 |

Run each at process affinities corresponding to 1, 6, and 12 logical CPU
threads. Q1--Q4 are qualification and sizing inputs only and never enter the
decision. E1--E6 are not used to tune the adapter, schedule, thresholds,
optimizer, or repeat count.

### Endpoints and observations

For every implementation/case/thread lane:

- three untimed full-fit warmups;
- 30 paired, order-randomized full-fit observations in fresh worker
  processes, seed `0xD0EB1008`;
- no slow-observation exclusion; one whole-lane rerun is allowed only for an
  explicit interruption or failure, and a second failure halts the lane;
- record total fit time and the existing objective-setup, initial-parameter,
  optimization, and parameter-extraction sections;
- record iterations, function/gradient evaluations, convergence, objective,
  full fitted state checksum, selected DVEB schedule, library load time,
  context scratch, process RSS, affinity, timestamps, and execution order.

Separately report resident fused objective-call overhead, artifact/library
load-to-first-answer, artifact size/dependencies, and a torch-blocked
load-to-fit measurement. These operational endpoints do not replace the
full-fit decision endpoint.

Use medians of the 30 observations. Report p05, p95, median absolute
deviation, and paired bootstrap 95% intervals with 10,000 resamples. Intervals
are descriptive; the frozen median rules decide.

### Correctness admission during the campaign

Both implementations must converge to an admitted solution in every timed
observation. Per lane, final `loglik`, `muhat`, and `sigmahat` must satisfy the
existing PyStatistics comparison tolerances and the fit must pass the existing
degeneracy guard. An observation with a numerical failure is evidence, not an
outlier to discard. Unequal optimizer iteration or evaluation counts are
reported and are part of end-to-end behavior.

### Frozen non-domination decision

Define lane speedup as `PyTorch time / DVEB time` for the full user-visible
fit. DVEB is dominated if either:

1. any lane has median `PyTorch/DVEB <= 0.10` (an order-of-magnitude
   embarrassment); or
2. PyTorch is faster in at least 12 of 18 lanes and the geometric mean of
   `DVEB/PyTorch` over all lanes is at least `2.0`.

If neither condition holds and every functional gate passes, the result is
GO. `PyTorch/DVEB >= 1.5` is a preferred, nonmandatory win and is reported per
lane. Being faster is preferred but not required; correctness, torch-free
deployment, and freedom from domination are the load-bearing claims.

## Interpretation limits

A GO establishes only that this explicit CPU-only implementation is suitable
for continued branch research. It does not make DVEB the default, remove
PyTorch, establish cross-platform wheel support, validate CUDA dense algebra,
change any paper result, or establish superiority for other statistical
workloads. A release or default-routing decision requires a later product and
packaging protocol after the publication freeze.
