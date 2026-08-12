# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **MICE MPS fast path: retired the merge-rank bridge on torch ≥ 2.13**
  (`mice/backends/_gpu_methods.py::_insertion_rank`). PyTorch 2.13's
  native-Metal `searchsorted` removed the pathology the ~10-op
  sort+cumsum+scatter bridge worked around (measured 2026-08-12, M2 Max:
  native searchsorted is 9.5–16.4x faster than the bridge at MICE shapes,
  m=100, n=2k–20k). Version-gated via the new
  `core.compute.device.mps_native_kernels()` predicate — torch ≤ 2.12 keeps
  the bridge unchanged. Results identical up to the documented ±1 tie
  convention, which the exact contiguous-block window search already refines
  away.
- **MICE MPS fast path: matmul-series inverse now used at every size on
  torch ≥ 2.13** (`mice/backends/_gpu_linreg.py`). The
  `_SERIES_INV_MIN_NOBS = 3000` crossover was tuned against the torch
  2.10/2.11 MPS dispatch floor (~0.5–1 ms/op); with that floor collapsed
  (~4 µs/op on 2.13), re-ablation at the same shapes (m=100, q+1=21) shows
  the series inverse beats `solve_triangular` at EVERY n_obs — 6.75x at
  n_obs=1700, 3.2x at 6800, 1.95x at 17000 — so on torch ≥ 2.13 the
  threshold is bypassed (always series on MPS). Older torch keeps the tuned
  threshold. CUDA/CPU unchanged (`solve_triangular`, fast there).
- **New capability predicate `core.compute.device.mps_native_kernels()`** —
  true when the installed torch is ≥ 2.13 (the release whose native-Metal op
  migration made `searchsorted`/`scatter`/`sort`/reductions fast on MPS).
  Lazy torch import; shared by the two MPS fast-path gates above.
- **Net effect (measured, M2 Max, torch 2.13, warm min-of-3 A/B on
  identical data):** MICE PMM (p=20, m=100, maxit=5) end-to-end GPU time
  0.92 → 0.70 s at n=2,000 (1.31x), 0.88 → 0.79 s at n=8,000 (1.11x),
  2.04 → 1.88 s at n=20,000 (1.09x). Gains concentrate at small n (the
  dispatch-bound corner), as the acceleration audit predicted.
- **MVNMLE MPS-EM refusal: corrected the stated justification**
  (`mvnmle/solvers.py`). The old message blamed Metal kernel-launch overhead —
  stale now that the dispatch floor collapsed. Re-measurement with the refusal
  bypassed (2026-08-12, direct `EMBackend('mps')`): fp32 EM on wine (178x13)
  needed 1002 iterations vs the fp64 CPU's 21 (11.1 s vs 41 ms — the
  fixed-point stalls at the fp32 noise floor), and on breast_cancer (569x30)
  it fails outright (E-step Cholesky loses positive-definiteness in fp32).
  The refusal is behaviorally correct and stays; the message now gives the
  real (precision) reason. No behavior change.
- **Fixed `benchmarks/mvnmle_bench.py` broken against the 4.0+ API** — the script
  still passed the pre-4.0 `algorithm=` kwarg to `mlest()` (renamed to `method=`
  in the 4.0 consistency release), so every case failed with `TypeError`; and its
  `_gpu_available()` checked only `torch.cuda.is_available()`, silently skipping
  all GPU rows on Apple Silicon. Both fixed (`method=` at the two call sites; MPS
  now detected). Discovered during the torch 2.13 re-baseline — the script is not
  exercised by CI, which is how it rotted unnoticed since 4.0.
- **docs: corrected hardware attribution in `docs/GPU_NOTES.md` and
  `docs/ACCELERATION_AUDIT_2026-08.md`** — the Apple Silicon benchmark
  machine was always a Mac Studio **M2 Max** ("Mainframe", since renamed
  "Powerhouse"; the fleet never included an M2 Ultra). All "M2 Ultra"
  chip references (both the April 2026 numbers and the 2026-08-12
  re-measurements) were misprints; numbers are unchanged. Since the two
  measurement campaigns ran on the same physical machine, hardware is a
  controlled variable and the scatter_add_ speedup is attributable purely
  to the software stack.
- **docs: corrected stale Apple-GPU (MPS) performance claims in
  `docs/GPU_NOTES.md`** after a torch 2.13.0 re-measurement (M2 Max vs RTX
  5070 Ti, 2026-08-12): `scatter_add_` with sparse targets is now 0.45 ms per
  2.5M-item call (was ~150 ms; 7.6x CUDA, ~8x faster than CPU `np.add.at`) and
  `searchsorted` at n=20k is 37 µs (was ~1136x slower than CUDA) — both fixed by
  PyTorch 2.13's native-Metal op migration; the "~0.5–1 ms per-op dispatch floor,
  not recoverable, no graph capture" claim is corrected to the measured
  decomposition (encode ~1–6 µs; commit+sync ~0.1–1 ms; torch 2.12+ eager
  ~4.2 µs/op; `torch.compile` works on MPS at ~0.023 µs/op). `solve_triangular`,
  `linalg.solve`/`inv`, and the newly-implemented-but-slow `cholesky_solve`
  remain 100–3300x off (existing matmul-series workarounds stay justified);
  `eigh`/`lstsq` remain unimplemented. Op-status table now carries both torch
  versions.
- **docs: corrected `pystatistics/CONVENTIONS.md` GPU non-target rationale** —
  the "inherently sequential algorithms without exploitable parallelism (… Cox
  PH partial likelihood)" bullet now states the accurate, threshold-based
  reason (Cox risk-set sums are prefix sums, GPU-proven at n ≳ 1e5;
  Kalman/additive-ETS admit associative scans with measured crossover T ≈ 1e6).
  No decision changes — only the stated reason.
- **Fixed a stale doc pointer in the MVNMLE MPS-EM refusal message**
  (`pystatistics/mvnmle/solvers.py`): the error referred users to
  `docs/GPU_BACKEND_NOTES.md`, which does not exist — corrected to
  `docs/GPU_NOTES.md`. (Note: this refusal's stated reason — Metal
  kernel-launch overhead — is itself a Phase-1 re-evaluation candidate per
  the acceleration audit, since the measured MPS dispatch floor collapsed
  on torch ≥2.12.)
- **docs: added `docs/ACCELERATION_AUDIT_2026-08.md`** — the full lower-level
  acceleration viability audit (Mojo/Triton/raw CUDA/Metal verdicts, torch 2.13
  re-benchmark, and the phased upgrade path for the MICE/MVNMLE GPU backends).
- **Validation note:** full test suite run on torch 2.13.0 — CUDA (Linux,
  RTX 5070 Ti): 4407 passed / 0 failed / 92 skipped. macOS arm64 (MPS,
  M2 Max): 4389 passed / 109 skipped / 1 failed —
  `tests/timeseries/test_ets_selection.py::TestDampedConvergence::
  test_free_phi_dominates_fixed_phi_probes`, a CPU-only, deterministic
  platform-libm optimizer-trajectory sensitivity (free-phi ETS(M,Ad,M) fit
  lands 0.25 loglik below the fixed-phi probe on macOS; passes on Linux;
  reproduced identically across numpy 2.4.6/2.5.2 and scipy 1.17.1/1.18.0, so
  neither torch- nor dependency-related — the same failure class as the
  Windows platform-libm case in `docs/CYTHON_MIGRATION_PROPOSAL.md` §17).
  Not fixed here: the test asserts a dominance property the optimizer genuinely
  misses on this platform, and whether to robustify the test or the phi-probe
  logic is a design decision to make deliberately. (Since fixed — see the
  ETS damped-fit phi-cascade entry below; the diagnosis also sharpened: the
  same Mac passes with an OpenBLAS/conda stack, so it is FP-stack sensitivity
  generally, not Apple libm specifically.)
- **Fixed ETS free-phi damped fits landing in a worse basin than fixed-phi
  fits** (`timeseries/_ets_fit.py::fit_ets_model`) — the macOS failure of
  `test_free_phi_dominates_fixed_phi_probes` above. On Accelerate-BLAS pip
  builds, both existing phi starts (0.9 and R's 0.9782) converged co2
  ETS(M,Ad,M) to loglik -66.944 while the same model with phi *fixed* at 0.98
  reached -66.692 — impossible at a true optimum, since the free-phi feasible
  set contains every fixed-phi model. Added a third optimiser leg: a
  pin-and-release cascade that first optimises with phi pinned at the 0.98
  upper bound (where both this failure and the 4.6.2 stall lived — damping
  degenerates toward "no damping" there and the logit transform saturates),
  then restarts the free-phi optimisation from that optimum with phi nudged
  1e-6 inside the bound. L-BFGS-B accepts only descent steps, so the free fit
  now matches any phi=0.98 fit to within ~3e-4 loglik units by construction,
  on every platform. A strict better-than tie-break keeps the plain-start
  result unless the cascade genuinely wins, so healthy fits are unchanged:
  across all seven damped R-reference fits in both macOS envs, only the
  broken co2 fit moved (-66.944 → -66.690, now strictly beating the fixed
  probe) plus a ~3e-6 improvement on wwwusage AAdN; no R-parity fixture
  shifts. Cost: one extra pinned fit (one dimension smaller) plus one
  near-converged release run, paid only by free-phi damped fits (measured
  on co2 MAdM: 0.89 s → 1.07 s, +21%). Test left untouched — it now
  certifies real dominance again.
- **Refactor: ETS initial-state code moved to `timeseries/_ets_init.py`** —
  `_init_level_trend`, `_init_season`, and `_assemble_init_states` moved
  verbatim out of `_ets_fit.py` (which had crossed the 400-LoC soft limit
  after the phi-cascade fix above; now 392 LoC) into a new focused module.
  Pure move, no behavior change: damped reference fits (co2 MAdM,
  airpassengers MAdM, wwwusage AAdN) verified bit-identical, full
  timeseries suite green before and after with identical counts.
