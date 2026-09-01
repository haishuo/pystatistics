# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **`mvnmle` now warns before a survey-scale MPS fit on a torch build carrying
  the allocator-state matmul misread.** MICE has been guarded since 6.1.3 and
  MVNMLE was not, although both run on the same backend against the same torch
  builds and `mvnmle`'s FP32 objective forms `sigma = L @ L.T`
  (`mvnmle/_objectives/gpu_fp32.py`) -- a transposed, therefore strided,
  operand, which is exactly the shape the upstream bug misreads. On an affected
  build (torch <= 2.13.x, per `core.compute.device.mps_misread_status`), a
  `DirectMLEBackend.solve` on MPS at n >= 20,000 now raises a `UserWarning`
  naming the operation and every remedy: `backend='cpu'`, CUDA, or torch >= 2.14.
  It is a warning rather than an error for the reason MICE gives -- moderate-n
  MPS results are validated and a user may knowingly accept the risk -- and it
  is silent on mitigated builds, below the threshold, and on CUDA and CPU. The
  threshold is asserted equal to MICE's, so the two guards cannot drift apart.
  No numerical behaviour changes on any path.

- **Apple-Silicon bootstrap of the mean now runs on ALLOY-generated Metal.**
  `boot(data, statistic, backend='gpu', gpu_statistic='mean')` on an
  Apple-Silicon machine routes the eligible design (ordinary method,
  `statistic_type='index'`, no strata, 1-D data) to a packaged ALLOY artifact
  instead of PyTorch. `backend_name` reports `gpu_mps_alloy_bootstrap`, and
  `result.info` carries `implementation='alloy'` with the ALLOY commit the
  binaries were built from. CUDA is unchanged and still uses PyTorch; the CPU
  path is untouched.

  What this buys a user: the complete `boot(...)` call is 1.10-1.72x faster
  than the PyTorch/MPS path across n=100..10,000 and 2,000..100,000 resamples
  (medians of 10 timed runs in alternating order,
  `benchmarks/montecarlo/bench_alloy_bootstrap.py`, Apple M2 Max). A fixed seed
  now reproduces the replicate means *exactly* on repeated runs, and the drawn
  indices are identical on the CPU and the GPU — the PyTorch path documents its
  GPU stream as statistically equivalent to the CPU's but not identical.

  Unsupported configurations are unchanged: a declared statistic that is not
  the mean still raises, and balanced/parametric bootstrap, frequency and
  weight statistics, strata and 2-D data are still refused on the GPU.

- **`backend='gpu'` no longer requires PyTorch on Apple Silicon** for the
  eligible bootstrap-of-the-mean. That request is now decided before the shared
  backend resolver, which reaches for PyTorch only to learn that a Metal device
  exists -- something ALLOY's own runtime reports directly. The resolver itself
  is unchanged, so `backend='auto'` still resolves to the CPU on Apple Silicon
  and CUDA still routes through PyTorch. An unsupported bootstrap configuration
  keeps its existing `ValidationError` and its existing wording -- on a machine
  without PyTorch installed as well as on one with it. If the packaged ALLOY
  backend cannot run, the call fails with that reason rather than quietly using
  PyTorch or the CPU.

- **New package data: a vendored ALLOY runtime and bootstrap bundle** under
  `pystatistics/montecarlo/backends/_alloy/artifacts/` (macOS arm64 only,
  ~348 KB). They are loaded from the installed package and from nowhere else —
  no environment variable, no source-tree search, no system location — and
  every file's SHA-256 is verified before use, so a partial or corrupted
  install fails by name rather than crashing in a dynamic loader.
  `PROVENANCE.md` beside them records the ALLOY commit, compiler and
  bundle-format versions, source and binary digests, and the rebuild command.
  On any other platform the loader raises with the reason and nothing else is
  affected.

  **PyStatistics remains MIT-licensed. The vendored ALLOY runtime and
  generated artifacts are separately identified and distributed under
  Apache-2.0**, with the notice Apache-2.0 requires shipped beside them as
  `LICENSE-ALLOY`. The MIT licence at the repository root does not extend to
  those files, and their terms do not extend to anything else.
  **These artifacts are not yet redistributable**: ALLOY's license is
  undecided, so they must not ship in a released wheel or sdist until that is
  settled. See the LICENSING section of that `PROVENANCE.md`.
- Ported the 6.1.2 logreg dtype-aware line-search slack to the GPU polyreg and
  polr fits (`_halving_step` in `pystatistics/mice/backends/_gpu_polyreg.py`,
  `_backtracking_step` in `pystatistics/mice/backends/_gpu_polr.py`): the
  acceptance slack was a fixed `1e-8 * (1 + |f0|)`, appropriate for fp64 but far
  below the fp32 penalised-NLL evaluation noise at survey n — measured on MPS at
  n=50k, K=3, m=20: noise sd ~8e-6 / max ~2.3e-5 vs slack ~5e-7 (17–47x). Near
  convergence the descent test then failed on pure noise: 798 of 2766 full-step
  rejections across 30 survey-scale polyreg sweeps (and 13 of 36 on an n=50k
  ordered polr fixture) were of steps fp64 confirms descend — zero after the
  fix. The slack is now `max(1e-8, 100*eps)` of the compute dtype (fp64
  behaviour unchanged). Impact: polyreg's line search poisons the chain on
  no-descent, so a noise-exhausted halving budget would refuse a whole run —
  the exact GSS n=50k logreg failure mode, closed here before any
  polyreg-bearing survey at that scale hits it; polr chains no longer risk
  freezing at a pre-convergence iterate on noise. Spurious rejections also
  burned ~2.8x the line-search NLL evaluations — the fixed polyreg fit ran
  ~1.5x faster at n=50k on MPS (43.8s → 30.0s over 30 sweeps), polr ~1.7x
  (1.2s → 0.7s per fit). Fits are calibration-identical: both slacks land on
  the same optimum (max deviation from the fp64 reference 7e-4 vs 6e-4).
  Regression tests: `TestLineSearchSlackDtypeAware` in
  `tests/mice/test_gpu_glm_separation.py` pins the mechanism deterministically
  for polyreg, polr, and the originating logreg fix (noise-scale gap must
  accept the full step; a genuine increase must still reject).

- Hardened the shared batched Gram/Hessian factor `_cholesky_ridged`
  (`pystatistics/mice/backends/_gpu_linreg.py`, used by the numeric
  norm/pmm draw and the polyreg Newton): it previously added one fixed
  `1e-8·scale` jitter and returned `cholesky_ex`'s output without ever
  reading `info` — on a failed factorization that output can be FINITE
  garbage (measured on MPS at n=50k: 5 of 10 failed polyreg-Hessian factors
  were finite), which flowed silently into the Newton step and the posterior
  draw. It now (a) escalates the jitter per chain ×100 up to `1e-2·scale`
  (the logreg `_pd_gram_cholesky` calibration, covering genuine fp32
  accumulation-error indefiniteness at survey n), implemented as a fixed
  speculative unroll so the hot numeric path stays free of per-step host
  syncs, with first-try chains keeping their factor bit-identically; and
  (b) explicitly returns an all-NaN factor for a chain that never reaches
  PD, so genuinely wrong matrices reach the backend's end-of-sweep guard
  instead of degrading silently. Fixed the companion silent path in
  `gpu_pmm_impute` (`pystatistics/mice/backends/_gpu_methods.py`): donor
  copies are finite observed values, so a non-finite predicted mean
  previously produced a silently scrambled but fully finite imputation
  (measured); rows/chains with non-finite predictions now impute NaN (the
  `_sample_categories` fail-loud pattern), with donors and RNG consumption
  unchanged for finite rows. Regression tests:
  `TestCholeskyRidgedEscalationAndPoison` and `TestPmmFailLoud` in
  `tests/mice/test_gpu_glm_separation.py`.

- KNOWN ISSUE documented (not fixed here — needs a decision): torch MPS
  (verified on 2.7.1–2.12.1 AND current stable 2.13.0) can return
  strided-matmul results wrong by 10–30% of scale when an operand lands in
  certain caching-allocator buffer placements after prior large allocations;
  tensor contents verify intact (the exact corrupting tensors replay clean
  in a fresh process), a fresh process computes correctly. At n=50k, m=20
  this fits ~1 polyreg chain in 20 to a silently wrong, finite optimum
  (measured max|beta| 49.6 vs 15.6 CPU fp64/fp32 ground truth) that no
  non-finite guard can see, and it may explain the 6.1.2 logreg "per-chain
  Gram cholesky coin flip" on GSS (unconfirmed — the isolated logreg fit
  does not corrupt; needs a real-GSS rerun). Follow-up retest results:
  torch 2.13.0 is NOT safe (the minimal repro goes quiet but the
  production-shaped fixture corrupts identically); the corruption stops
  reproducing at pytorch nightly 2.14.0.dev20260624 — bisected by dated
  nightlies to pytorch #187441 (2026-06-23, an MPS caching-allocator
  bucketing change that is a memory optimization, NOT a kernel fix, so the
  bug is MASKED rather than fixed; the 2.13 branch was cut 2026-06-10,
  before it, and the 2.14 branch, cut 2026-08-11, contains it — 2.14.0
  stable should be clean). Nightly 2.15.0.dev20260813 verified clean
  (production fixture 16/16, full mice suite passes). The
  `.contiguous()`-LHS mitigation is insufficient (the misread relocates to
  other strided ops). The NaN-poisoning factor converts the detectable
  (indefinite-Hessian) half into loud refusals; the remaining PD-but-wrong
  fits are covered by the warning + canary (next bullet). Full analysis in
  `docs/GPU_NOTES.md` ("MPS strided-matmul buffer corruption"). Reported
  upstream as pytorch/pytorch#193487 (2026-08-13, self-contained
  fire-and-forget report). Re-validation of the published v6.1.2 GSS/CSES
  MPS survey legs (2026-08-14, record in
  `pystatistics-validation/artifacts/mice/v6.1.2/runs/`): NO corruption
  fingerprint — torch 2.13.0 and the mitigated nightly are statistically
  indistinguishable against a torch-free CPU reference on every cell,
  consistent with those cells never invoking polyreg (the corruption-prone
  shapes); the published numbers stand.

- Added the three-layer defense for the MPS misread: (1)
  `mps_misread_status()` in `pystatistics/core/compute/device.py` classifies
  the installed torch as 'affected' (<= 2.13.x, or 2.14 nightlies before
  dev20260624) or 'mitigated' (2.14 line and later), conservative on
  unparseable versions; (2) the MICE GPU backend
  (`pystatistics/mice/backends/gpu.py`, `_warn_if_mps_misread`) emits a
  UserWarning before `backend='gpu'` runs on MPS at n >= 20,000 under an
  'affected' torch — silent wrong results must be loud at the point of
  use — and stays silent on a 'mitigated' torch, so installing one (e.g. a
  torch nightly, the documented no-promises opt-in) is the supported path
  to a quiet run; (3) new
  `pystatistics/mice/diagnostics.py::mps_matmul_canary()` fits a fixed
  survey-scale fixture on MPS and CPU (fp32) and returns
  'corrupted'/'clean' in ~3 s (measured: deviation 42.9 on torch 2.12.1 vs
  1.1e-5 on the clean nightly — six orders of magnitude of separation),
  wired into `.release/CHECKLIST.md` as step 0 for Mac releases. README
  gains a user-facing "Known issue" section with the remedies (CPU backend,
  CUDA, torch >= 2.14 / nightly >= 2.14.0.dev20260624, canary). Tests:
  `tests/mice/test_mps_misread.py` (classifier version table, warning
  fire/silence matrix including a real threshold-lowered MPS mice run,
  canary verdict sharpness).

---

## Research line: `research/alloy-pystatistics`

This branch is the ALLOY research line. It is **not** a release line and must
never be tagged, released, or merged to `main` as-is.

- Base: `v6.1.5` (`d68f011`), the exact tagged release published to PyPI.
- Version identity `6.2.0.dev0+alloy` — deliberately not `6.1.5`, so an
  installed copy can never be mistaken for the released package.
- Carries the vendored ALLOY runtime (Apache-2.0; see
  `pystatistics/montecarlo/backends/_alloy/PROVENANCE.md`). PyStatistics itself
  remains MIT.
- The macOS wheel floor is 13.0 here because the vendored ALLOY dylibs require
  it. That floor belongs to this branch only.
