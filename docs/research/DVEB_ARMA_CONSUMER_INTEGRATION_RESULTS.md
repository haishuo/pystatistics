# DVEB exact-ARMA consumer integration result

**Date:** 2026-09-02

**Outcome:** **PHASE-I PASS / PHASE-II PENDING.**

The frozen DVEB trunk-009 artifact can be consumed through a private,
research-only PyStatistics adapter with exact artifact identity checks,
truthful CPU and transfer-inclusive CUDA modes, complete fixed-parameter
correctness qualification, and no change to released behavior.  This result
does **not** establish an exact batch fitter or authorize a public backend.

## Scope and publication firewall

All work is confined to `research/dveb-pystatistics`, whose research version
is `6.2.0.dev0+dveb`.  PyStatistics `main`, tag/PyPI version 6.1.5, the
MVN-MLE paper and replication materials, ALLOY, DVEB, GradFlow, PyStatsBio,
and the public `arima_batch` contract are unchanged.  In particular,
`arima_batch(..., method="ml")` still refuses; no new fit route, optimizer,
default, release, tag, merge, or package upload exists.

## Frozen artifact identity

The adapter consumes the external handoff produced from DVEB commit
`7955be7c3c6f0c6ed976d963d669a8823a17de4a`, tree
`2295a6172a7055d38547db055056da5b82e7c627`:

- handoff archive SHA-256:
  `33886fbbfc54028dbb99964d0c01251fd413c79eff76064afbe95a0a558999dd`;
- handoff manifest SHA-256:
  `dcff39c98c6d3442ff3a255e55ba2fa90a705dbfc028e2739abcbe53d18e07bb`;
- CPU-only ABI SHA-256:
  `3349732f1dc3e9d3f65ca32581028c49e7e870d4369b072f1f05b6ce46fdd428`;
- combined CPU/CUDA ABI SHA-256:
  `328893657d4688a703b2661f1bb9facfc141b95369a02f5fc62e93e889c984f6`;
- language-source SHA-256:
  `faf72d2a406decbd65b68d1d404a90cbffebf98594b8bc93b132f71cd66b0ea9`;
  and
- ABI version 1, contiguous row-major float64, `items >= 1`, `steps >= 1`,
  and `1 <= state <= 25`.

Only the CPU-only shared library is bundled under the private package.  ELF
inspection found no CUDA, NVIDIA, RPATH, or RUNPATH dependency.  Its admitted
platform is Linux x86-64-v2, which the loader checks before loading.  The
combined CUDA artifact is deliberately not bundled: CUDA use requires the
caller to supply its exact external path, whose hash is checked before load.

## Adapter boundary

`pystatistics.timeseries._dveb_arma` exposes two private research components:

- `DVEBCPUExactArma` retains a CPU context and accepts only explicit forced
  serial or item-parallel schedules.  It rejects DVEB CPU automatic placement
  because trunk 009 did not qualify that selector on held-out shapes.
- `DVEBCudaTransferExactArma` retains a CUDA context and truthfully invokes the
  handoff's host upload/run/download helper.  It supports forced blocks
  32/64/128/256 and the frozen calibrated automatic mapping only for admitted
  state extents.  It never describes this path as device-resident.

Both accept already-prepared `(z, phi, loading)` NumPy arrays and return
`(nll, sigma2, status)`.  They do not own AR/MA parameter mapping,
differencing, means, seasonal expansion, fitting, convergence, covariance,
warnings, or result construction.  Inputs must already be float64,
two-dimensional, C-contiguous, shape-compatible, and non-aliased under the ABI
contract; no implicit copy or fallback repairs an invalid call.

## Correctness qualification

The independent authority is PyStatistics's existing Cython/NumPy exact-ARMA
implementation and the frozen Phase-0B evidence.  Qualification passed:

- all 16 frozen grid cells under 8 schedules each: CPU serial at one thread,
  CPU item-parallel at 6 and 12 threads, CUDA forced 32/64/128/256, and CUDA
  automatic;
- persistent-stationary, diffuse/nonstationary, and invalid/non-finite status
  cases under all 8 schedules;
- all eight primary proposal traces (`E04`, `E05`, `E06`, `E07`, `E08`,
  `E09`, `E11`, `E12`), 100 proposals each under all 8 schedules: 6,400
  schedule-proposal evaluations, zero numerical failures, and zero status
  mismatches;
- caller inputs unchanged, CPU schedules bitwise identical, and fixed-policy
  results deterministic;
- worst frozen-cell absolute differences of 3.8198777474462986e-11 for NLL
  and 7.993605777301127e-15 for sigma2, both within the prospectively frozen
  scale-aware bounds; and
- worst proposal-trace relative differences of 4.239584344587835e-15 for NLL
  and 1.1542645010098784e-14 for sigma2.

Focused refusal tests passed for automatic CPU scheduling, missing and
tampered artifacts, wrong container/dtype/layout/shape, illegal thread and
CUDA-block controls, closed contexts, missing explicit CUDA artifact, and the
unchanged public exact-ML refusal.  The CPU adapter also executed successfully
in a subprocess that actively prohibited importing `torch`; `torch` never
entered that process.

## Regression firewall

The corrected frozen regression campaign passed:

- time-series suite: 1,182 passed, 13 deselected;
- archived Phase-0B evidence verifier: PASS;
- Phase-I consumer-evidence verifier: PASS; and
- complete suite: 4,485 passed, 94 skipped, 27 deselected, with no new failure
  relative to frozen pre-adapter commit `f156ee5`.

The complete suite reproduced two inherited failures.  The multinomial
complete-separation failure was already recorded at `a9c7e4d`.  The GPU
kurtosis test was reproduced at `f156ee5` before the corrected restart with
the identical arrays and identical 6.79605395e-07 discrepancy.  The first
regression attempt and that baseline reproduction are preserved; neither test
was changed or waived.

## Interpretation and next gate

Phase I establishes a safe, private fixed-parameter likelihood component and
a genuine torch-free CPU deployment path.  It inherits the previously frozen
Phase-0B performance authority; no adapter timing was performed here.

The current ABI returns likelihood values but no parameter derivatives and
owns no optimizer.  Consequently it cannot yet establish that a complete
exact-ML fit is correct, non-dominated, or operationally preferable.  Phase II
must remain separately authorized and prospectively frozen.  Its first screen
must compare the existing Cython/SciPy fitter, competent ordinary PyTorch, and
a PyStatistics-owned finite-difference construction over the admitted DVEB
likelihood without changing estimator semantics.  If repeated likelihood
calls are structurally dominant despite correct fits, the honest result is a
DVEB capability request for general derivative emission—not an ARMA-specific
compiler exception.

## Evidence

- protocol: `docs/research/DVEB_ARMA_CONSUMER_INTEGRATION_PROTOCOL.md`;
- machine-readable qualification:
  `docs/research/evidence/dveb_arma_consumer/qualification.json`;
- machine-readable regression:
  `docs/research/evidence/dveb_arma_consumer/regression.json`;
- preserved process attempts:
  `docs/research/evidence/dveb_arma_consumer/PROCESS_ATTEMPTS.md`; and
- offline verifier: `benchmarks/dveb_arma_consumer/verify_evidence.py`.
