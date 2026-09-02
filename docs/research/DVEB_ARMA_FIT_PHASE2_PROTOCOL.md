# DVEB exact-ARMA Phase-II fit-suitability protocol

**Date:** 2026-09-02

**Status:** **APPROVED AND FROZEN before any Phase-II fit implementation,
calibration, evaluation, or timing.**  This document was drafted after Phase I
completed and approved on 2026-09-02 before any Phase-II fit implementation,
calibration, correctness comparison, or timing.

**Research branch:** `research/dveb-pystatistics`

## 1. Binary question

Phase I qualified a private PyStatistics adapter for the frozen DVEB
fixed-parameter exact-ARMA likelihood.  Phase II asks one narrower question:

> Can that immutable likelihood artifact support a statistically equivalent,
> operationally bounded, and non-dominated many-series exact-ML fit without a
> new DVEB compiler feature?

DVEB need not beat every PyTorch or Cython lane.  It must not be embarrassed:
an order-of-magnitude loss is catastrophic and systematic material domination
is NO-GO.  A speed win is preferred and earns a descriptive STRONG GO, but it
is not required for the base performance gate.

The unit under test is a **complete private batch fit**, not a likelihood call
or a prerecorded proposal trace.  It begins with frozen host-resident series
and start arrays and ends with host-resident fitted coefficients, profile
sigma2, convergence/failure state, iteration counts, and a result checksum.

## 2. Nonclaims and publication firewall

- Work remains only on `research/dveb-pystatistics`, version
  `6.2.0.dev0+dveb`.
- `main`, tag/PyPI 6.1.5, the MVN-MLE paper and replication materials,
  ALLOY, DVEB, GradFlow, PyStatsBio, and released behavior remain untouched.
- No merge, release, tag, package upload, or public documentation change is
  authorized.
- `arima_batch(..., method="ml")` continues to refuse.  Phase II uses a
  private research harness and cannot itself authorize a public backend.
- No DVEB compiler change is authorized.  A missing derivative or resident
  bridge is recorded as a possible general DVEB capability, never implemented
  as a PyStatistics/ARMA compiler exception in this repository.
- This first fit screen does not claim seasonal, differenced, mean-bearing,
  exogenous-regressor, missing-observation, forecasting, covariance/Hessian,
  or device-resident Python support.

## 3. Immutable foundations

### 3.1 DVEB handoff

Use the exact Phase-I artifacts and loader without modification:

- DVEB source commit
  `7955be7c3c6f0c6ed976d963d669a8823a17de4a`, tree
  `2295a6172a7055d38547db055056da5b82e7c627`;
- CPU ABI SHA-256
  `3349732f1dc3e9d3f65ca32581028c49e7e870d4369b072f1f05b6ce46fdd428`;
- combined ABI SHA-256
  `328893657d4688a703b2661f1bb9facfc141b95369a02f5fc62e93e889c984f6`;
- ABI version 1, float64, contiguous row-major arrays, state 1 through 25;
- CPU forced serial/item-parallel schedules only; and
- CUDA forced 32/64/128/256 plus the Phase-I frozen automatic mapping
  `r=1 -> 32`, `r=3 -> 32`, `r=13 -> 256`, `r=25 -> 256`.

The CPU artifact remains the only bundled library.  CUDA requires the exact
external combined artifact.  No fallback, recompile, or artifact search occurs
during a fit.

### 3.2 Statistical authority

The authority is the existing PyStatistics exact Gaussian ARMA likelihood:
stationary covariance by the 60-iteration relative-tolerance doubling method,
diffuse fallback with `kappa=1e6`, companion-state Kalman recurrence, and
profile sigma2.  Final candidate vectors are always re-evaluated by the
existing Cython/NumPy path; a candidate may not validate itself.

The first screen is non-seasonal `d=0`, `include_mean=False`, no `xreg`, with
fixed-zero masks for sparse high-state families.  This is the largest subset
the frozen ABI can fit without rebuilding `z` for a changing mean or adding
consumer mathematics unrelated to the question.

### 3.3 Environment

Use the already installed Phase-0B environment:

- `/mnt/projects/dveb/.venv-torch/bin/python`;
- Python 3.12.3;
- NumPy 2.5.2;
- SciPy 1.18.1;
- PyTorch 2.13.0+cu130 and Triton 3.7.1;
- CUDA runtime 13.0 on the Forge RTX 5070 Ti, compute capability 12.0.

Freeze the complete environment report and wheel hashes before implementation.
Do not substitute the separate `gradflow` environment, whose PyTorch build is
a 2.9.0 development build with CUDA 12.8.

## 4. Parameter families

The data bytes and family coefficients are the already-frozen Phase-0B
`F1`, `F3`, `F13`, and `F25` records.  The fit parameter vector contains only
the entries below; every unlisted AR/MA coefficient is fixed to exact zero.

| Family | State/order representation | Free vector, in order | Truth |
|---|---|---|---|
| F1 | ARMA(1,0), `r=1` | `ar1` | `(0.60)` |
| F3 | ARMA(3,2), `r=3` | `ar1, ar2, ar3, ma1, ma2` | `(0.50,-0.20,0.05,0.30,-0.10)` |
| F13 | fixed-sparse ARMA(13,1), `r=13` | `ar1, ar12, ar13, ma1` | `(0.40,0.45,-0.18,0.20)` |
| F25 | fixed-sparse ARMA(25,1), `r=25` | `ar1, ar24, ar25, ma1` | `(0.25,0.20,-0.05,-0.25)` |

The sparse families are legitimate fixed-parameter ARMA models, not a new
seasonal optimizer: the effective coefficient arrays are passed directly to
the same likelihood.  The mask, expansion, and MA sign convention are shared
by every implementation and unit-tested independently.

### 4.1 Starting values

Starts are generated once before implementation and committed with hashes:

1. compute the existing biased Yule-Walker start separately for each series
   at the full AR order;
2. retain only the family's free AR lags and set fixed lags to exact zero;
3. clip each retained value to `[-0.99,0.99]`;
4. if the reconstructed AR polynomial is not strictly stationary with minimum
   root modulus at least 1.001, repeatedly multiply all free AR values by 0.5
   until it is, with a hard limit of 64 halvings and refusal on exhaustion; and
5. initialize every free MA coefficient to exact zero.

No truth coefficient enters the start.  All routes receive byte-identical
starts.  Start generation is correctness-only and excluded from steady fit
timing, but included in the fresh-process endpoint.

## 5. Frozen grid and roles

### 5.1 Calibration — never decision evidence

The disjoint Phase-0B calibration cells select only the finite-difference and
ordinary-PyTorch variants described below:

| Cell | K | n | Family |
|---|---:|---:|---|
| C01 | 8 | 120 | F1 |
| C02 | 64 | 500 | F3 |
| C03 | 256 | 2,000 | F13 |
| C04 | 1,024 | 500 | F25 |

Calibration results may not establish GO/NO-GO and are not pooled with
evaluation.  Variant identities and hashes are frozen immediately after
selection and before any evaluation cell is executed.

### 5.2 Evaluation

| Cell | K | n | Family | Role |
|---|---:|---:|---|---|
| E01 | 1 | 120 | F1 | characterization: single short fit |
| E02 | 1 | 2,000 | F13 | characterization: single high-state fit |
| E03 | 16 | 500 | F3 | characterization: small batch |
| **E04** | **64** | **2,000** | **F3** | **primary** |
| **E05** | **256** | **500** | **F13** | **primary** |
| **E06** | **256** | **2,000** | **F13** | **primary** |
| **E07** | **1,024** | **2,000** | **F3** | **primary** |
| **E08** | **1,024** | **500** | **F13** | **primary** |
| **E09** | **64** | **10,000** | **F3** | **primary** |
| E10 | 16 | 2,000 | F25 | characterization: small high-state batch |
| **E11** | **256** | **500** | **F25** | **primary** |
| **E12** | **1,024** | **120** | **F1** | **primary: many short fits** |

The eight bold cells are the complete decision class.  E01/E02/E03/E10 are
reported but cannot establish or veto a decision except through correctness.
No grid extension or cell deletion is allowed after evaluation begins.  An
OOM, timeout, compiler refusal, or resource failure at its original cell is a
result.

### 5.3 Public-data correctness-only case

Add one application check from the committed R fixture: take
`log(AirPassengers)`, apply one ordinary difference, subtract its sample mean,
and fit ARMA(1,1) with no fitted mean.  Run K=1 and a K=32 batch of exact byte
copies.  This case checks the private fitting path on public data but is never
timed and does not imply R parity for a model configuration R did not record.

## 6. Optimizer semantics and batching

Every fit is an independent SciPy L-BFGS-B instance with the same start and:

- `ftol=1e-8`;
- `gtol=1e-5`;
- `maxiter=300`;
- `maxfun=15000`;
- `maxcor=10`;
- `maxls=20`; and
- no coefficient bounds.

These preserve the current PyStatistics optimizer family and the public batch
iteration budget.  Supplying a gradient changes only how the gradient is
obtained; it does not authorize a joint optimizer, Adam, SPSA, or a shared
stopping rule.

### 6.1 Deterministic independent-optimizer coordinator

The many-series routes use a private coordinator solely to batch backend
requests while preserving one optimizer state per series:

1. at most 256 independent optimizer workers are active at once; larger K is
   processed in consecutive input-order chunks of 256;
2. each active optimizer submits one `(f,g)` request and blocks;
3. the coordinator waits until every still-active optimizer has either
   submitted its next request or reported completion/failure;
4. it evaluates one backend batch containing one request per active series,
   returns each row only to its owning optimizer, and repeats; and
5. completion removes that optimizer from subsequent barriers without
   changing another optimizer's state.

There is no time-window coalescing, queue timeout, parameter sharing, summed
objective, global line search, or cross-series Hessian update.  Batch grouping
therefore cannot change an optimizer's `(f,g)` sequence.  Thread count, chunk
bound, thread-stack size, and request ordering are fixed before calibration.
If this coordinator cannot reproduce independent reference fits, Phase II
halts; it is not replaced after evaluation by a joint optimizer.

## 7. Implementations

| ID | Implementation | Gradient | Role |
|---|---|---|---|
| S0 | existing Cython likelihood + independent SciPy L-BFGS-B, up to 12 concurrent CPU fits | SciPy-compatible forward difference | operational CPU incumbent and correctness authority |
| TC1 | ordinary eager PyTorch public-op exact recurrence, coordinator, CPU float64 | PyTorch autograd | CPU PyTorch candidate |
| TC2 | `torch.compile(fullgraph=True)` structured-while recurrence, coordinator, CPU float64 | compiled PyTorch autograd | CPU compiler candidate if admitted |
| TG1 | ordinary eager PyTorch public-op exact recurrence, coordinator, CUDA float64 | PyTorch autograd | CUDA PyTorch candidate |
| TG2 | `torch.compile(fullgraph=True)` structured-while recurrence, coordinator, CUDA float64 | compiled PyTorch autograd | CUDA compiler candidate if admitted |
| D1 | Phase-I DVEB CPU-only adapter, coordinator, 12 threads | selected batched finite difference | CPU-only candidate |
| D2 | Phase-I DVEB `cuda-transfer` adapter, coordinator, calibrated block mapping | selected batched finite difference | NVIDIA candidate under the current ABI |

S0 is a private batch harness over the current single-series Cython likelihood
and SciPy optimizer, not a claim that PyStatistics already ships an exact
`arima_batch` route.  It uses the current SciPy absolute forward step
`eps=1e-8`.  Its parallelism is a persistent 12-thread pool over independent
fits, the strongest ordinary CPU use of the existing compiled likelihood;
K=1 uses one worker.

TC1/TC2/TG1/TG2 may use only public PyTorch operations, autograd, and
`torch.compile`.  CPU routes pin PyTorch intra-op threads to 12 and inter-op
threads to one; CUDA routes keep the same coordinator CPU limits and execute
the numerical recurrence on the GPU.
No custom CUDA, Triton, C++ extension, generated-kernel edit, private compiler
hook, graph break, or eager fallback under a compiled name is allowed.  The
data and parameters may remain resident for the fit; final results are copied
to host.  Compilation is excluded from the steady endpoint and reported
separately, then included in the fresh-process endpoint.

D1 uses forced item-parallel scheduling at 12 threads for every primary cell;
the K=1 characterization uses forced serial.  D2 uses only the already-frozen
automatic CUDA block mapping.  Forced schedules remain correctness controls,
not post-result tuning.  Because the current CUDA adapter is explicitly
transfer-inclusive, every ABI call's actual uploads/downloads are counted and
reported.  No result may call it resident.

### 7.1 DVEB finite-difference selection

Two choices are declared before implementation:

- **FD-F:** forward difference with absolute `h=1e-8`, one base plus one
  positive perturbation per free parameter, dividing by the actual
  representable displacement `(x+h)-x`; and
- **FD-C:** central difference with absolute `h=1e-5`, the step used by the
  existing public `arima_gradient` helper, with positive and negative
  perturbations per free parameter plus the unperturbed objective, dividing
  by the actual displacement `(x+h)-(x-h)`.

For one coordinator request, perturbations from all active series are expanded
into one contiguous likelihood batch.  Make separate D1 and D2 selections so
the CPU-only and CUDA conclusions do not force one device's cost model onto the
other.  On each device, both choices must pass every calibration correctness
gate.  If only one passes, select it for that route.  If both pass, run 10
complete-fit calibration repetitions per C01--C04 and select the lower
geometric mean of the four per-cell median wall times.  A relative difference
below 2% is a tie and selects FD-F because it requires fewer likelihood rows.
Freeze exactly one scheme and its hashes for D1 and one for D2 before
evaluation.  No hybrid by family or cell is permitted.

### 7.2 PyTorch selections

Make separate CPU and CUDA selections.  TC1/TC2 and TG1/TG2 must independently
pass calibration correctness on their respective device.  If a compiled route
refuses or breaks its full graph, its eager peer survives and the refusal is
preserved.  For each device where both pass, use the same 10 calibration
repetitions and geometric-mean rule; a difference below 2% selects the eager
route because it has no compilation dependency.  Freeze exactly one ordinary-
PyTorch CPU variant and one ordinary-PyTorch CUDA variant before evaluation.
Unselected variants remain disclosed calibration results and cannot be
substituted later.

## 8. Correctness admission

No implementation/cell is timed until all applicable gates pass.

### 8.1 Likelihood and gradient

At starts, 20 deterministic interior perturbations per family, and every final
fit vector:

- status agrees exactly with Cython/NumPy;
- candidate NLL and sigma2 agree with independent Cython re-evaluation under
  the Phase-0B bound
  `max(32*eps*n*max(1,r), 1e-14) * max(1,abs(reference))` separately for each
  output;
- D1 and D2 finite-difference gradients agree with the same formula evaluated
  through Cython at `rtol=1e-5, atol=1e-5`; and
- PyTorch autograd agrees with a Cython central-difference audit at
  `rtol=1e-4, atol=1e-6`.

The gradient bounds are deliberately looser than the likelihood bound because
subtraction divides independent-compiler roundoff by the finite-difference
step.  They are frozen before results.  A failed gradient scheme is rejected;
the tolerance is not relaxed.

### 8.2 Complete fit

For every series in every calibration/evaluation/public-data cell:

- success/failure masks agree exactly with S0;
- successful outputs are finite and final AR polynomials are stationary;
- MA coefficients are normalized through the existing invertible-
  representative helper before comparison;
- maximum absolute difference from S0 is at most `5e-3` for every free AR/MA
  coefficient;
- absolute difference in independently re-evaluated NLL is at most `0.05`;
- sigma2 agrees at `rtol=1e-4, atol=1e-8`;
- the candidate-reported final NLL agrees with independent Cython evaluation
  under the stricter fixed-likelihood bound above;
- caller data and start arrays are byte-unchanged;
- fixed-zero coefficients remain bitwise zero; and
- iteration count, callback count, likelihood-row count, line-search status,
  and final gradient norm are recorded (but need not equal across derivative
  methods).

Every candidate final result is also re-run from the same start to prove fixed-
policy determinism.  Persistent-stationary, diffuse/nonstationary, invalid,
partial-failure, and all-failed batches are qualification cases.  No failed
row may be silently replaced by S0 or another backend.

### 8.3 Regression and deployment

- all Phase-I adapter tests and evidence verifiers pass;
- all existing time-series tests pass;
- the complete suite introduces no failure relative to frozen commit
  `f156ee5` and its two proven inherited failures;
- importing `torch` is actively blocked while a D1 complete fit executes;
- the bundled CPU artifact again shows no CUDA/NVIDIA/RPATH/RUNPATH dependency;
  and
- public `arima_batch(..., method="ml")` remains unchanged and refusing.

## 9. Timed endpoints

### F1 — persistent complete fit (decision endpoint)

Start with frozen host `z` and start arrays.  Include parameter expansion,
coordinator/optimizer work, every likelihood and gradient evaluation, required
transfers/synchronization, MA normalization, final independent-validity check,
host result construction, and checksum.  Exclude input/start generation,
imports, artifact loading, CUDA context creation, and PyTorch compilation.
Contexts and compiled graphs are already warm and persist across observations.

F1 is the only endpoint used for CPU and NVIDIA non-domination decisions.

### F2 — fresh-process launch to answer (descriptive)

At E07 only, include interpreter launch, imports, artifact/framework load,
input and start generation with hash checks, context/graph creation and
compilation, one complete fit, host serialization/checksum, and shutdown.
Measure peak RSS.  Thirty independent processes per admitted implementation
are required.  F2 cannot rescue or overturn F1; it bounds deployment cost.

### F3 — resources and work accounting (descriptive)

Report CPU peak RSS, GPU peak allocated/reserved memory, artifact/import
footprint, compile/load time, optimizer iterations, `(f,g)` callbacks,
likelihood rows, ABI calls, input bytes uploaded, output bytes downloaded,
kernel launches where traceable without privileged counters, and partial-batch
utilization as optimizers finish.  Profiling runs are separate from timed runs.

## 10. Timing design

- Forge is exclusive for each GPU block; CPU affinity and governor are
  recorded and unchanged.
- One block is one implementation/cell in a fresh worker process.
- Six worker processes each perform one untimed complete-fit warmup followed by
  five timed fits: exactly 30 F1 observations per implementation/cell.
- Worker-block order is randomized from a committed seed, stratified so no
  implementation always runs first or last.
- One F1 observation is one complete fit; no in-region repetition multiplier
  is allowed.
- F2 uses 30 fresh processes per implementation at E07.
- Use monotonic wall time around the complete declared region and synchronize
  the GPU only at its boundaries or where the truthful adapter contract
  requires it.
- Median of all 30 observations is the headline.  Report worker medians, p05,
  p95, MAD, and all raw observations.
- Pair implementations by cell and worker index.  Use a committed bootstrap
  seed and 10,000 hierarchical paired resamples (workers first, then the five
  observations within each selected worker) for 95% ratio intervals.
- Compilation, calibration, correctness, profiling, and failed attempts never
  enter F1.

A calibration-only planning pass estimates the campaign before evaluation.  If
any cell estimate misses by more than 10x or total projected evaluation time
exceeds eight hours, stop for a **duration-only** review.  Grid, thresholds,
correctness rules, implementation identities, and decision rules remain
immutable; no automatic sample-count reduction is authorized.

## 11. Thermal, failure, and exclusion rules

Use the CUDA-13 toolkit profiler paths and `nvidia-smi`; do not use the stale
system profiler binaries or request privileged counters.

Exclude an observation only for a recorded external event: a non-idle throttle
reason, SM clock below 90% of that worker's warmup median, temperature at least
80 C, a competing GPU process, worker crash, or OS-level interruption.
Numerical failure, nonconvergence, OOM, graph refusal/break, unsupported shape,
or adapter refusal is never excluded.

If more than 20% of a worker block is externally excluded, rerun the entire
block once.  A second failure marks it unstable; retained samples are not
cherry-picked.  Cool to at most 55 C or wait 60 seconds between GPU worker
blocks.  Do not lock clocks or modify Forge.

## 12. Prospective decisions

Ratios are `incumbent median / DVEB median`; above one favors DVEB.  The best
existing comparator is the faster admitted existing implementation for that
platform and cell, never an average.  Intervals must exclude 1.0 for a cell to
count as a speed win; domination counts use medians and the fixed material
thresholds below.

### 12.1 CPU-only result

Compare D1 with the faster of S0 and the calibration-selected ordinary-
PyTorch CPU route on every primary cell.  PyTorch's distribution size is
reported separately and does not remove it as a performance comparator.

- **Catastrophic NO-GO:** any primary ratio `best_cpu_existing / D1 <= 0.10`.
- **Systematic-domination NO-GO:** the existing CPU envelope is faster in at
  least 6 of 8 primary cells and geometric-mean
  `D1 / best_cpu_existing >= 2.0`.
- **GO:** every correctness/deployment gate passes and neither NO-GO rule
  fires.
- **STRONG GO (descriptive):** D1 is faster in at least 6 of 8 primary cells
  and geometric-mean `best_cpu_existing / D1 >= 1.25`.

This result supports only a forced-schedule, Linux x86-64-v2, torch-free CPU
research route.  It makes no CUDA claim.

### 12.2 NVIDIA result

Compare D2 with the faster of S0, the selected ordinary-PyTorch CPU route, and
the selected ordinary-PyTorch CUDA route on each primary cell.

- **Catastrophic NO-GO:** any primary ratio `best_existing / D2 <= 0.10`.
- **Systematic-domination NO-GO:** the existing envelope is faster in at least
  6 of 8 primary cells and geometric-mean `D2 / best_existing >= 2.0`.
- **GO:** every correctness/deployment gate passes and neither NO-GO rule
  fires.
- **STRONG GO (descriptive):** D2 is faster in at least 6 of 8 primary cells
  and geometric-mean `best_existing / D2 >= 1.25`.

The per-cell existing envelope deliberately gives DVEB the harder test.  D2's
result applies only to the disclosed transfer-inclusive ABI; it cannot be
reported as device-resident.

### 12.3 Overall interpretation

- **Fit GO:** at least one of CPU-only or NVIDIA is GO.  Only the passing
  explicit route may proceed to a separately reviewed private integration.
- **Fit STRONG GO:** descriptive only, attached independently to each route.
- **NO-GO / Case 1:** both routes are correctly expressible but systematically
  dominated by existing implementations without evidence that the frozen ABI
  boundary caused the loss.  Preserve the result and add no exact batch
  backend.
- **HALT / Case 2:** fixed likelihood remains promising, but complete fitting
  fails correctness or non-domination for an attributable general missing
  capability—principally derivative emission or a reusable owned/device-
  resident parameter/data bridge.  Preserve the exact evidence and return to
  DVEB only under a new general compiler/runtime protocol.

A CPU GO cannot conceal a CUDA NO-GO, or vice versa.  Both are printed.  A
correctness failure supersedes performance.  Deployment value alone cannot
waive catastrophic or systematic domination.

## 13. Attribution requirements

Before drawing a language-level conclusion, attribute any material distance
among S0, PyTorch CPU, PyTorch CUDA, D1, and D2 to at least one measured
boundary:

- likelihood rows required by finite differences versus autograd;
- optimizer/coordinator time outside numerical kernels;
- batch occupancy decay as independent fits finish;
- CPU schedule and thread utilization;
- repeated host/device transfers and synchronization;
- PyTorch compilation, graph structure, and forward/backward launches;
- generated DVEB resource use and launch policy; or
- result validation/materialization.

“DVEB is slow” is not an adequate diagnosis if the code written on top of the
artifact, the emitted artifact, or the ABI boundary is the actual cause.

## 14. Freeze, execution order, and stop conditions

After approval, work proceeds in this exact order:

1. commit this protocol and a machine-readable declaration that no Phase-II
   comparative results exist;
2. commit input/start generator, hashes, coordinator design tests, and
   implementation identities;
3. implement S0/TC1/TC2/TG1/TG2/D1/D2 without timing evaluation cells;
4. pass calibration correctness, select FD and PyTorch variants using only
   C01--C04, and commit the selection/hashes;
5. pass exhaustive evaluation/public-data correctness and the full regression
   firewall;
6. commit the admitted-campaign manifest before timing;
7. run F1, then F2/F3, preserve every raw observation, and apply the decision
   mechanically;
8. commit the result and stop with a clean worktree.

Do not optimize after seeing evaluation results.  Do not alter a tolerance,
grid point, route, chunk bound, threshold, or result because a decision is
unfavorable.  Do not push later commits without new explicit authorization.

Approval of this completed protocol authorizes the private Phase-II
implementation, correctness qualification, calibration selection, and—only
after a separate admitted-campaign freeze—the 30-observation campaign.  It
does not authorize a public API, DVEB compiler work, merge, release, tag,
package upload, or push.
