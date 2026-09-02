# DVEB exact-ARMA consumer integration and fit-suitability protocol

**Date:** 2026-09-02

**Status:** APPROVED AND FROZEN before adapter implementation or timing.
Approval on 2026-09-02 authorizes Phase-I private-adapter implementation and
correctness qualification only. It does not authorize Phase-II fit work, a
public API, comparative timing, merge, release, or package upload.

**Branch:** `research/dveb-pystatistics`

## 1. Question and boundary

Trunk 009 established that DVEB can emit a correct fixed-parameter exact-ARMA
likelihood for CPU and CUDA and that its CUDA artifact is not dominated by the
strongest admitted ordinary-PyTorch implementation. This protocol asks the
next two questions in order:

1. can PyStatistics consume that frozen artifact through a narrow, truthful,
   research-only adapter without changing released behavior; and
2. does the artifact provide a credible route from fast likelihood evaluation
   to an actual batched exact-ML fit?

The questions are not conflated. A correct fast likelihood is useful evidence,
but it is not itself a fitter. The current artifact returns per-series NLL,
sigma2, and status; it does not return parameter derivatives and it does not
own an optimizer. `arima_batch` currently supports only `method='whittle'`.
No result may describe exact batch fitting as implemented until the fit gate in
this protocol is separately satisfied.

## 2. Publication firewall

- Work occurs only on `research/dveb-pystatistics`, version
  `6.2.0.dev0+dveb`, descended from the frozen PyPI 6.1.5 line.
- `main`, tag `v6.1.5`, PyPI, the MVN-MLE paper and replication material,
  ALLOY, GradFlow, PyStatsBio, and released behavior remain unchanged.
- Nothing is merged to `main`, tagged, released, or uploaded to a package
  index during the publication freeze.
- Branch pushes require explicit authorization. A branch push is not a merge
  or release.
- DVEB compiler changes belong only in the DVEB repository. PyStatistics owns
  the adapter, model parameterization, optimizer, result contract, and product
  policy; DVEB must not acquire PyStatistics- or ARMA-specific compiler rules.

## 3. Frozen handoff

The consumer input is the external bundle
`/mnt/artifacts/dveb/trunk009_exact_arma_20260902`, produced from clean DVEB
commit `7955be7c3c6f0c6ed976d963d669a8823a17de4a`, tree
`2295a6172a7055d38547db055056da5b82e7c627`.

- archive SHA-256:
  `33886fbbfc54028dbb99964d0c01251fd413c79eff76064afbe95a0a558999dd`;
- bundle manifest SHA-256:
  `dcff39c98c6d3442ff3a255e55ba2fa90a705dbfc028e2739abcbe53d18e07bb`;
- combined ABI SHA-256:
  `328893657d4688a703b2661f1bb9facfc141b95369a02f5fc62e93e889c984f6`;
- CPU-only ABI SHA-256:
  `3349732f1dc3e9d3f65ca32581028c49e7e870d4369b072f1f05b6ce46fdd428`;
- language-source SHA-256:
  `faf72d2a406decbd65b68d1d404a90cbffebf98594b8bc93b132f71cd66b0ea9`;
- ABI version: 1; and
- admitted shape: contiguous row-major float64, `items >= 1`, `steps >= 1`,
  `1 <= state <= 25`.

Every copied artifact is verified against `SHA256SUMS` before it is loaded.
The adapter uses an exact package-relative path and rejects a hash, ABI,
platform, dtype, shape, layout, alias, schedule, or device mismatch. It never
searches arbitrary library paths and never silently falls back.

The CPU-only library is `x86-64-v2` and has no CUDA/NVIDIA/PyTorch dependency.
The combined artifact is qualified only on CUDA 13.0 and compute capability
12.0. Neither identity implies broader Linux or GPU portability.

## 4. Phase I: private fixed-parameter adapter

Phase I adds no public fitting method. It provides one private research
component accepting already-prepared `(z, phi, loading)` arrays and returning
the artifact's `(nll, sigma2, status)` arrays.

The adapter has two explicit modes:

- `cpu`: load the CPU-only artifact, create and retain one context, and invoke
  the CPU ABI with an explicit forced serial or item-parallel schedule;
- `cuda-transfer`: load the combined artifact and use the disclosed host
  upload/run/download helper with the frozen calibrated block model.

CPU automatic scheduling is not admitted because the DVEB held-out evaluation
preserved three material selector mistakes. Phase I must select a forced CPU
schedule from a predeclared consumer rule or explicit test control. CUDA
automatic scheduling is admitted with its recorded E08 8.52% miss; forced
blocks remain available for qualification.

The initial adapter does not pretend that `cuda-transfer` is resident. A later
resident Python route must either consume caller-owned CUDA tensors explicitly
or receive separate authorization for a general owned-buffer bridge. It may
not hide allocation, transfer, or synchronization.

### Phase-I correctness gates

Before any adapter timing:

1. verify archive, member hashes, ABI version, required symbols, CPU ISA, ELF
   dependencies, and CUDA eligibility;
2. reproduce all 16 Phase-0B frozen cells and the complete 100-proposal traces
   against the existing Cython/NumPy authority under the already-frozen
   scale-aware tolerance;
3. exercise CPU forced serial/item-parallel schedules at 1, 6, and 12 threads
   and CUDA forced blocks 32, 64, 128, and 256 plus calibrated automatic;
4. preserve exact status, finite-output, diffuse, persistent-root, non-finite,
   wrong-shape, wrong-dtype, noncontiguous, alias, closed-context, illegal
   schedule, and unavailable-device behavior;
5. prove caller inputs unchanged and repeated results deterministic for a
   fixed policy;
6. prove CPU results bitwise identical across legal schedules;
7. prove a subprocess in which importing `torch` raises can load and execute
   the CPU-only adapter; and
8. pass the focused time-series suite and introduce no new failure in the
   complete default suite relative to frozen commit `f156ee5`.

No failed candidate is timed. No tolerance is changed after comparative data
exist.

### Phase-I interpretation

Passing Phase I establishes a safe private likelihood component and torch-free
CPU deployment. It does not establish an exact batch fitter, public API,
optimizer equivalence, default routing, packaging portability, or release
readiness. The prior DVEB campaign remains the performance authority for the
fixed-parameter engine; Phase I does not rerun it merely to obtain a consumer
wrapper number.

## 5. Phase II: fit-suitability screen

Phase II begins only after Phase I passes. It freezes a separate fit protocol
before implementing or timing an optimizer. Its purpose is to decide whether
the admitted likelihood can support a statistically and operationally honest
batched exact-ML route without first requiring an unbounded new language.

### Required parameter mapping

The consumer, not DVEB, owns mapping candidate AR/MA parameters to the
effective companion `phi` and noise-loading arrays, differencing, centering or
mean handling, stationarity/invertibility policy, starts, convergence, Hessian
or covariance estimation, warnings, and `ARMABatchSolution` construction.
Seasonal factored parameters must be expanded by the existing audited
polynomial helpers. Unsupported cases refuse rather than silently selecting
Whittle or a single-series CPU loop.

The first fit screen is limited to non-seasonal `d=0` ARMA families with
`include_mean=False`. This is an explicit research subset, chosen because the
frozen artifact takes a fixed `z` array and has no in-kernel fitted-mean
parameter. Mean fitting, differencing, seasonal factorization, exogenous
regressors, missing observations, forecasting, and covariance/Hessian
construction are later gates, not assumed consequences.

### Candidate fit routes

All candidates use the same initial parameters, parameter constraints,
stopping rule, iteration cap, exact likelihood, and final validity checks.

1. **Cython/SciPy incumbent:** the existing exact single-series Cython
   likelihood in independent SciPy L-BFGS-B fits. This is the correctness and
   operational baseline, not a straw man.
2. **Ordinary PyTorch:** the public-op exact recurrence with
   `torch.while_loop`, float64, and PyTorch automatic differentiation where
   supported. A read-only feasibility probe before this draft confirmed that
   both the for-loop and structured-while formulations can produce gradients;
   this is no timing evidence. No custom CUDA, Triton, extension, generated
   code edit, or private compiler hook is allowed.
3. **DVEB finite-difference route:** central or the existing SciPy-compatible
   forward finite differences formed by the consumer from multiple admitted
   DVEB likelihood calls. Probe batches are formed over parameter directions
   without changing the estimator. The difference scheme and step rule must be
   frozen from existing SciPy/PyStatistics behavior before evaluation.

SPSA, a new Adam estimator, approximate/Whittle substitution, mixed precision,
associative time scans, and workload-specific CUDA/Triton are excluded from
the first screen because they would confound backend and optimizer changes.

If no DVEB finite-difference construction can reproduce the incumbent fits
within the frozen statistical contract, Phase II is NO-GO for the current
handoff. If derivatives are correct but performance is structurally dominated
by repeated likelihood calls, record Case 2: a derivative-producing general
DVEB capability may be proposed in a later compiler trunk, but is not
authorized here.

### Fit qualification set

Freeze disjoint calibration and evaluation generators before implementation.
At minimum they include AR(1), ARMA(3,2), and one higher-state sparse ARMA
family; small, ordinary, and many-series batches; short and long time axes;
persistent stationary, diffuse/nonstationary, invalid, and partial-failure
cases; and at least one public-data fixture already admitted by PyStatistics.

For every admitted evaluation lane, require:

- all implementations receive byte-identical data and starts;
- convergence/status and failure masks agree;
- final NLL/log-likelihood agrees under a prospectively derived scale-aware
  bound;
- fitted AR/MA coefficients and sigma2 agree under existing PyStatistics fit
  tolerances;
- stationarity/invertibility conclusions agree;
- optimizer iterations and objective/gradient evaluations are reported;
- the final DVEB result is re-evaluated by the independent Cython authority;
  and
- no input mutation, hidden fallback, or undisclosed transfer occurs.

### Performance rule to freeze before timing

The fit campaign retains DVEB's governing non-domination principle. Define
lane speedup as `best_existing / DVEB` for the complete fit, including
parameter mapping, derivative construction, optimization, final validation,
and result materialization.

- catastrophic NO-GO if any primary lane has median
  `best_existing / DVEB <= 0.10`;
- systematic-domination NO-GO if the best existing route is faster in at
  least 75% of primary lanes and geometric-mean `DVEB / best_existing >= 2.0`;
- otherwise performance is GO if every correctness/deployment gate passes;
- STRONG GO is descriptive and requires DVEB faster in at least 75% of primary
  lanes with geometric-mean `best_existing / DVEB >= 1.25`.

The exact grid, primary class, warmups, 30-observation design, randomization,
bootstrap seed, duration bounds, thermal rules, and failure/exclusion rules
must be added and separately approved before fit implementation. No threshold
or evaluation point may be revised after comparative results exist.

## 6. Possible outcomes

- **Phase-I PASS / Phase-II pending:** qualified private fixed-parameter
  adapter only; the accurate description until a fitter is admitted.
- **Fit GO:** current handoff supports a correct non-dominated bounded exact
  batch fit and may proceed to a separately reviewed research-only public API.
- **Fit STRONG GO:** the same, with the preferred performance result.
- **NO-GO / Case 1:** ordinary PyTorch or the existing Cython/SciPy route covers
  the fit cleanly and materially dominates the current handoff; preserve the
  result and do not add a DVEB public backend.
- **HALT / Case 2:** the workload remains promising but needs a general missing
  DVEB capability such as derivative emission or a reusable owned-device
  buffer bridge. Record the exact capability and return to the DVEB repository
  only under separate authorization.

## 7. Stop and authorization gates

Approval of this protocol authorizes Phase-I adapter implementation and
correctness qualification only. It would not authorize fit implementation,
fit timing, a new public `arima_batch` option, a DVEB compiler change, copying
the combined CUDA library into a default CPU package, a merge, release, tag,
or package upload.

After Phase I, stop with a clean research worktree and report the artifact
identity, adapter boundary, every correctness/refusal result, torch-free CPU
result, regression result, and any missing ABI capability. Phase II requires a
second explicit approval of its completed grid and measurement protocol.
