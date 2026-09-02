# DVEB batched exact-ARMA Phase-0B suitability protocol

**Date:** 2026-09-02

**Status:** APPROVED AND FROZEN before implementation or timing. The initial
approval authorized the consumer-side NumPy/Cython/PyTorch workload/null
preflight. A separate approval on 2026-09-02 authorizes the NC/NG diagnostics
and the prospective native-headroom campaign without changing this protocol's
grid, endpoints, thresholds, or decision rules.

**Branch:** `research/dveb-pystatistics`

## 1. Question and stopping rule

This Phase-0B asks whether many independent exact Gaussian ARMA Kalman
likelihoods expose native CPU/CUDA headroom that is not already recovered by a
competent Cython or ordinary-PyTorch implementation.

It is a workload/capability screen, not a backend implementation. The result is
binary:

- **Case 1 / NO-GO:** existing ordinary tools cover the production-relevant
  region without material domination or the native route has no credible path
  to end-to-end fitting. Preserve the evidence and do not create a DVEB trunk.
- **Case 2:** a separately authored native implementation demonstrates useful
  headroom or feasibility, but DVEB lacks one or more general mechanisms needed
  to express it. Schedule only those mechanisms in the DVEB repository.
- **Suitability GO:** DVEB later expresses the admitted program, passes every
  correctness gate, is not dominated under section 12, and retains a credible
  path to exact batched fitting. This requires a second, separately frozen
  DVEB-inclusive campaign; Phase-0B alone cannot award it.

The screen stops after its Case-1/Case-2 decision. It does not add
`method="ml"` to `arima_batch`, change any default, or optimize after seeing
decision results.

## 2. Repository and publication boundary

- PyStatistics authority: this research branch, descended from PyPI 6.1.5.
- PyPI 6.1.5, `main`, its tag, and the paper replication environment remain
  unchanged.
- GradFlow, ALLOY, PyStatsBio, and released PyStatistics are read-only.
- A native diagnostic may be added only on this research branch after this
  protocol is approved and frozen.
- Any DVEB language/compiler work requires a separate `codex/` branch and
  separate authorization in `/mnt/projects/dveb`.
- Nothing is merged, tagged, released, or uploaded to a package index.

## 3. Frozen mathematical contract

The first screen evaluates a batch of fixed parameter proposals. For series
`k`, it computes the existing PyStatistics concentrated exact Gaussian
time-domain likelihood:

```text
r = max(p, q + 1), with r = 1 for white noise
a_0 = 0
P_0 = stationary covariance from the existing doubling iteration,
      or kappa I with kappa = 1e6 when that iteration refuses

v_t = z_t - a_t[0]
F_t = P_t[0, 0]
K_t = P_t[:, 0] / F_t
a_filtered = a_t + K_t v_t
P_filtered = P_t - K_t P_t[0, :]
(a_{t+1}, P_{t+1}) = companion-state propagation + R R'

sse = sum_t v_t^2 / F_t
sigma2 = sse / n
nll = 0.5 n log(2 pi sigma2) + 0.5 sum_t log(F_t) + 0.5 n
```

All paths use float64, identical caller-owned input values, the existing
companion-state formulation, the same 60-iteration stationary-init limit,
relative convergence threshold `1e-13`, diffuse constant `1e6`, invalid-state
rules, and concentrated reductions. There is no Whittle substitution,
mixed-precision state, altered initialization, associative time scan, hidden
CPU fallback, or changed statistical estimator.

The CPU bit-identity oracle remains the pure NumPy/Cython operation order with
contraction disabled. CUDA and independently compiled implementations use
prospectively derived tolerances from section 9; bit identity is not promised
across compilers or devices.

## 4. Implementations and roles

| ID | Implementation | Role |
|---|---|---|
| N | Existing pure NumPy recurrence | readable correctness authority; never a performance straw man |
| C1 | Existing fused Cython workspace, serial batch loop | shipped single-series native incumbent |
| C2 | Competent multicore Cython/native batch control | CPU comparator; independent series scheduled across fixed workers with one workspace per worker/series |
| T1 | Ordinary PyTorch eager, time loop over batch-resident tensors | framework baseline |
| T2 | `torch.compile(fullgraph=True)` of the same public tensor formulation | primary ordinary-PyTorch compiler route |
| T3 | Public `torch.while_loop` formulation, if it passes all gates | additional ordinary-PyTorch control-flow route; prototype status and warnings reported |
| NC | Separately authored CPU native diagnostic | optional native CPU ceiling |
| NG | Separately authored CUDA diagnostic | optional native CUDA ceiling |

The PyTorch formulations are written from section 3, not transcribed from a
diagnostic kernel. Custom CUDA, C++, Triton, extensions, private Inductor hooks,
or handwritten generated code are forbidden in T1--T3. Graph breaks,
recompiles, generated kernel counts, materialized state, transfers, and compile
time are recorded. If TorchInductor captures the recurrence efficiently, that
is valid Case-1 evidence.

The native diagnostics are not DVEB and cannot establish a DVEB GO. They exist
only to answer whether useful native headroom exists. Authorship order and
source hashes must be recorded before any DVEB-generated implementation is
inspected for this workload.

## 5. Inputs and parameter families

All generated inputs use NumPy `PCG64DXSM` with one committed master seed and a
documented `(phase, cell, replicate)` substream map. Arrays are C-contiguous
float64 and hashed before an implementation initializes PyTorch or a thread
pool. Every implementation receives byte-identical arrays. Timed workers verify
the hashes again after execution.

Four admitted stable families exercise increasing local-state size:

| Family | Effective shape | Construction |
|---|---:|---|
| F1 | `r=1` | stationary AR(1), coefficient 0.60, no MA term |
| F3 | `r=3` | stable ARMA(3,2) using a committed root-safe coefficient vector |
| F13 | `r=13` | multiplicative nonseasonal/seasonal AR structure expanded to the existing companion representation |
| F25 | `r=25` | higher seasonal state, root-checked with minimum AR-root modulus at least 1.05 |

The exact F3/F13/F25 coefficients are generated and committed with the protocol
freeze, after verification by the existing stationarity/root machinery. They
cannot be changed after comparator results are visible.

Correctness-only families additionally include:

- persistent but stationary roots with minimum modulus in `[1.0001, 1.01]`;
- nonstationary parameters that must select diffuse initialization;
- invalid/non-finite parameters that must return the declared status/penalty;
- zero-length, wrong-shape, non-contiguous, and local-capacity refusals; and
- existing R 4.5.2 fixture-bearing cases, including seasonal `r=13` paths.

No invalid or diffuse-only case is timed.

## 6. Disjoint grid

Calibration and evaluation use disjoint `(K, n, r)` cells. Calibration may
choose only predeclared block/thread schedules and repetition batching. It may
not change formulas, implementation source, primary cells, or thresholds.

### 6.1 Calibration cells

| ID | K | n | r | Purpose |
|---|---:|---:|---:|---|
| C01 | 8 | 120 | 1 | launch/call floor |
| C02 | 64 | 500 | 3 | ordinary low-order batch |
| C03 | 256 | 2,000 | 13 | seasonal, sustained work |
| C04 | 1,024 | 500 | 25 | high local-state pressure |

### 6.2 Evaluation cells

| ID | K | n | r | Class |
|---|---:|---:|---:|---|
| E01 | 1 | 120 | 1 | single-series CPU characterization |
| E02 | 1 | 2,000 | 13 | single-series CPU characterization |
| E03 | 16 | 500 | 3 | small batch |
| E04 | 64 | 2,000 | 3 | primary |
| E05 | 256 | 500 | 13 | primary |
| E06 | 256 | 2,000 | 13 | primary |
| E07 | 1,024 | 2,000 | 3 | primary; shipped batch scale |
| E08 | 1,024 | 500 | 13 | primary |
| E09 | 64 | 10,000 | 3 | primary; long series |
| E10 | 16 | 2,000 | 25 | high-state characterization |
| E11 | 256 | 500 | 25 | primary; high-state batch |
| E12 | 1,024 | 120 | 1 | primary; many short series |

The eight primary cells are E04--E09, E11, and E12. E01--E03 and E10 are
reported but cannot establish or veto the primary many-series decision except
through the catastrophic-regression rule in section 12. The grid is fixed; no
post-result upward extension is permitted. A resource failure is a result.

## 7. Endpoints

Measure and report these endpoints separately:

1. **L1 resident likelihood:** inputs and scratch already resident; return
   per-series NLL, sigma2, and status.
2. **L2 transfer-inclusive:** caller host arrays through device execution and
   result materialization; CPU paths use the identical host inputs.
3. **L3 fit-trace proxy:** evaluate a committed sequence of 100 valid parameter
   proposals per series with workspace reuse. This tests the use pattern an
   optimizer creates without claiming to be an optimizer.
4. **L4 fresh process:** import/load, one E07 likelihood batch, result
   materialization, shutdown, and peak RSS.
5. **Artifact preparation:** compile/build time and artifact size, excluded from
   steady-state timing but reported.

L1 decides native/framework execution. L2 and L4 decide whether transfers and
deployment erase it. L3 is the load-bearing credibility bridge to a future
end-to-end fit. None substitutes for another.

## 8. Correctness admission

Before timing, every candidate must pass:

1. exact CPU equality with the pure NumPy reference wherever section 3's
   operation order and contraction policy are shared;
2. exact Cython agreement for stationary initialization, recurrence state,
   local scratch mutation, status, SSE, and `sum(log(F))` on the existing
   parity fixtures;
3. R-fixture agreement through the existing reference/full-fit path;
4. per-series NLL, sigma2, and status agreement on every calibration and
   evaluation cell;
5. persistent-root, diffuse, invalid, non-finite, shape, stride, and capacity
   gates;
6. caller-input immutability and deterministic repeat results;
7. auto/forced schedule mathematical agreement; and
8. complete existing PyStatistics time-series tests plus the focused new
   qualification tests.

A failed candidate is not timed. No tolerance is changed after evaluation data
are observed.

## 9. Tolerance derivation

Tolerance derivation uses correctness-only inputs disjoint from C01--C04 and
E01--E12. For every output, record absolute and scale-normalized differences
from Cython and NumPy.

The frozen CUDA/independent-compiler tolerance is the larger of:

- `32 * eps * n * max(1, r)` times the relevant reported scale; and
- four times the maximum difference observed on the disjoint derivation set,

rounded upward to one significant decimal order. NLL and sigma2 receive
separate bounds. Status must agree exactly. The factor and derivation records
are frozen before evaluation. If the resulting tolerance would exceed `1e-9`
relative for finite valid outputs, the candidate is rejected for review rather
than admitted with a looser rule.

## 10. Timing design

- one exclusive Forge GPU campaign;
- CPU affinity fixed and recorded;
- CPU threads `{1, 6, 12}` for C2/NC/DVEB; serial C1 remains one thread;
- fixed CUDA candidates `{32, 64, 128, 256}` threads plus any separately
  declared one-block-per-series layout;
- hardware properties cached outside timed regions;
- 10 warmups per implementation/cell/endpoint;
- 30 independent timed observations;
- implementation order randomized within each observation from a committed
  seed;
- one worker process owns one complete block of observations, with block order
  interleaved across implementations;
- region repetition chosen on calibration data to target at least 100 ms and
  fixed identically for competitors doing the same work;
- median is the decision statistic; p05, p95, MAD, and 10,000-pair bootstrap
  95% intervals are reported;
- compilation and first-specialization costs are excluded from L1--L3 and
  reported under artifact preparation;
- every timed observation includes a correctness checksum and input hash.

If any planning estimate misses observed calibration time by more than 10x,
stop and re-review duration only. Grid and decision rules do not change.

## 11. Thermal, failure, and exclusion rules

Use the pinned CUDA-toolkit Nsight tools and `nvidia-smi`; do not use the stale
system profiler copies. Hardware counters requiring administrative changes are
out of scope.

Exclude an observation only for a recorded external event: non-idle throttle
reason, SM clock below 90% of that configuration's warmup median, GPU
temperature at least 80 C, competing GPU process, worker crash, or OS-level
interruption. Numerical failures are never excluded.

If more than 20% of a block is excluded, rerun the whole block once. If it
fails again, report the cell as unstable; do not cherry-pick retained samples.
Any OOM, compiler refusal, graph break that violates fullgraph, local-memory
spill, status mismatch, or unsupported shape is reported at its original cell.

## 12. Prospective decision rules

Ratios are always `competitor median / candidate median`, so values above one
favor the candidate.

### 12.1 Phase-0B native-headroom decision

Case 2 is warranted only if an admitted native diagnostic shows at least one
of:

1. median L3 speedup at least 1.5x over the best competent ordinary existing
   path in at least 6 of 8 primary cells, with bootstrap intervals excluding
   1.0 and no primary cell below 0.80x; or
2. an existing path cannot execute at least 6 primary cells within the memory
   and correctness contract while the native diagnostic can, and the failure
   is not merely an unavailable optional dependency.

If neither condition holds, the result is Case 1 / NO-GO and no DVEB compiler
trunk is authorized.

### 12.2 Later DVEB non-domination rule

If Case 2 later produces a DVEB handoff, the DVEB-inclusive campaign uses all
of these rules:

- catastrophic NO-GO if any primary L1 or L3 lane has
  `best_existing / DVEB <= 0.10`;
- systematic-domination NO-GO if the best existing path is faster in at least
  6 of 8 primary L3 cells and the geometric mean `DVEB / best_existing` is at
  least 2.0;
- otherwise the performance requirement is GO: DVEB is not dominated;
- **STRONG GO** is descriptive and requires DVEB faster in at least 6 of 8
  primary L3 cells with geometric mean `best_existing / DVEB >= 1.25`;
- any DVEB/native-ceiling distance is reported and attributed to source,
  emitted code, launch/schedule policy, spills, memory traffic, or artifact
  boundary before drawing a language-level conclusion.

In addition, L2 and L4 must not reverse a resident STRONG GO into a systematic
loss, and L3 must retain the same qualitative result as L1. Otherwise the
fixed-parameter result has no credible path to fitting and the final outcome is
NO-GO.

## 13. Evidence and integrity

Commit before timing:

- this approved protocol;
- exact input generators, fixed arrays, and hashes;
- comparator and diagnostic source hashes;
- compiler/library/environment identities;
- calibration choices and evaluation freeze;
- threshold derivation; and
- a machine-readable declaration that no decision observations exist.

Commit after the campaign:

- every raw observation and exclusion;
- mechanical analysis and bootstrap seeds;
- correctness, graph, resource, memory, launch, transfer, and compile records;
- the binary Case-1/Case-2 result; and
- process failures and negative evidence with equal prominence.

The report must distinguish algorithm, source formulation, generated code,
compiler scheduling, runtime boundary, and hardware effects. A loss may not be
attributed to “DVEB” until those layers are inspected.

## 14. Authorization gates

Approval on 2026-09-02 authorizes the consumer-side NumPy/Cython/PyTorch
workload/null preflight. A separate approval later that day authorizes the
separately authored NC and NG diagnostics, their correctness admission,
calibration, and the frozen L1--L4 native-headroom campaign. It does not
authorize:

- a DVEB compiler or runtime change;
- a DVEB statistical special case;
- a PyStatistics exact-batch backend;
- an end-to-end optimizer;
- a merge, release, tag, or package upload; or
- any modification to GradFlow, ALLOY, PyStatsBio, or `main`.
