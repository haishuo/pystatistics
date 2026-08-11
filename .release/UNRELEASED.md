# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

Correctness fixes across the `mvnmle` submodule:

- **Fixed silent int64 pattern-code overflow in `mvnmle/patterns.py`
  (`identify_missingness_patterns`) corrupting Little's MCAR test for
  p >= 65.** `2 ** np.arange(...)` wraps to 0 in int64 for bit positions >= 64
  (first failure exactly p = 65), silently merging distinct missingness
  patterns — `little_mcar_test` then returned wrong statistic/df/p-value with
  no warning (repro: df 69 vs true 207 at p = 70). Applied the same
  arbitrary-precision object-dtype path used by the earlier fix in
  `MLEObjectiveBase._apply_mysort` (the sibling occurrences had been missed).
  New regression tests: `tests/mvnmle/test_patterns_large_p.py`.
- **Added a never-jointly-observed variable-pair guard
  (`_degeneracy.check_pairwise_observation`, wired into `mlest`).** When two
  variables never appear observed in the same row, sigma[i,j] enters no
  likelihood term: the likelihood is exactly flat in it and the entry is
  unidentified — previously `mlest` returned an arbitrary starting-point
  artifact for it with `converged=True` and no warning. Now raises
  `SingularMatrixError` (warning + `converged=False` under `force=True`).
- **Fixed the FP32/FP64 torch objectives' diagonal jitter to be
  scale-relative.** The batched objective added an ABSOLUTE `eps` (1e-6 in
  FP32) to every observed diagonal, biasing every fitted variance by -eps
  absolutely — about -10% on data with variances near 1e-5, accepted silently
  by every guard. The jitter is now `eps * var_j` per variable (observed
  variances, constant during optimization so the closed-form gradient stays
  exact), making the bias ~ -eps relative at any data scale. Unit-scale data
  (including all published benchmarks, which standardize columns) is
  numerically unaffected. New test: `tests/mvnmle/test_jitter_scale.py`.
- **Fixed SQUAREM's PD safeguard contract in the EM backend (spurious
  warnings on `mlest(missvals, method='em')`).** `em.py` passed the
  user-level `regularize` policy into `squarem_step`'s trial-iterate check,
  which must RAISE for the alpha back-off to engage; under the default
  (`regularize=True`) the back-off was dead and wildly extrapolated trial
  covariances were ridged with data-scale ridges (0.46 and 0.07 on missvals)
  plus factually wrong warnings. Trial iterates now use a strict feasibility
  check (mirroring the torch SQUAREM variant); the user policy applies once
  to the accepted iterate — which also closes the gap where SQUAREM's
  stall/exhausted fallback iterates skipped `ensure_pd` entirely. The
  canonical `mlest(datasets.missvals, method='em')` now converges with zero
  warnings and identical estimates.
- **GPU EM now honors `regularize=False`.** `_run_em_loop_gpu` hard-coded
  `_ensure_pd(..., regularize=True)`, so the documented strict mode was
  silently ignored on CUDA (with a warning telling the user to pass the flag
  they had already passed). The flag is now threaded through.
- **`mlest` conflicting-argument guards (fail loud instead of silently
  overriding).** `mlest(..., backend='gpu', solver='reference')` silently ran
  the numpy CPU reference (and skipped backend validation entirely — even
  `backend='bogus'` passed); now raises `ValidationError`, as does an
  explicit GPU backend with `method='monotone'` (a CPU closed form).
- **`mlest(method='em', backend='gpu_fp64')` no longer rejected as "Unknown
  backend".** EM already computes in float64 on CUDA, so the documented
  'gpu_fp64' backend now routes like 'gpu' (CUDA required) instead of raising
  a factually false error.
- **`mlest` no longer clobbers the direct backends' precision-calibrated
  tolerance.** `_solve_direct` forced `tol=1e-5` into every backend, making
  `DirectMLEBackend`'s FP32 default (`gtol=1e-3`, matched to the ~2e-4 FP32
  per-observation gradient floor) unreachable dead code — FP32 fits were
  driven at an unmeetable 1e-5 and converged only via the ftol exit. `tol`
  is now forwarded only when user-specified (same handling as `method`);
  docstring updated, `regularize` documented (EM-only).
- **Removed a silent identity-covariance fallback in the R-exact reference
  objective (`_objectives/cpu.py`).** On `LinAlgError` during parameter
  extraction, `sigmahat` silently became `np.eye(p)`; it is now NaN-filled so
  the degeneracy guard rejects it loudly. The source-guard test that should
  have caught this pointed at the dead `_utils.py` and has been retargeted.
  Also: a doubly-failed finite-difference gradient component now raises
  `NumericalError` instead of being silently zeroed.
- **`GPUObjectiveFP32` no longer silently reroutes an explicit
  `device='mps'` request to CPU when MPS is unavailable** — it raises
  `RuntimeError`, matching the FP64 class.
- **Monotone closed-form MLE: fixed an off-by-one identifiability guard.**
  `n_k = k+1` observations exactly interpolate the k-predictor regression
  (zero residual variance, singular Sigma returned silently);
  the guard now requires `n_k >= k+2`, and a zero residual variance from
  exact linear dependence raises `ValidationError`.
- **`little_mcar_test` p-value now uses `chi2.sf`** instead of
  `1 - chi2.cdf` (keeps precision in the far tail; matches R's
  `pchisq(lower.tail=FALSE)`), and the test suite now pins statistic/df/
  p-value against the previously orphaned R-generated reference fixture
  (`tests/mvnmle/references/little_mcar_apple.json`) instead of asserting
  only finiteness.
- **EM non-convergence warning now reports the actual EM-step count**
  (SQUAREM cycles can overshoot `max_iter` by a few steps; the warning
  claimed "after {max_iter} iterations" while `n_iter` said otherwise).
- **Dead code removed:** `mvnmle/_utils.py` (zero callers; still carried the
  int64 overflow bug), `MatrixLogParameterization` /
  `get_parameterization` / `convert_parameters` (zero callers),
  `e_step_full_batched_np` and `compute_conditional_parameters_torch`
  (zero callers, untested).

Test-suite robustness:

- **De-flaked the QR-solve performance test**
  (`tests/core/test_qr_solve.py::TestSpeed`): its wall-clock comparison
  against `numpy.linalg.lstsq` was load-sensitive and failed intermittently
  on machine noise; the timing comparison is now robust (no library code
  changed).

`gam` fix:

- **Fixed `gam` smoothing-parameter branch resolution losing the accepted
  P-IRLS branch (`_criteria.select_lambdas`).** With a multimodal inner
  P-IRLS problem at near-zero penalty, branch resolution warm-started the
  final fit from the LAST criterion evaluation's fitted mean — but
  L-BFGS-B's rejected line-search trials after accepting the optimum still
  overwrite that state, and a far-away trial can hop the inner fixed point
  onto a different branch. The reported fit then sat on a branch the
  criterion was never accepted on (observed on
  `gam(..., family=Gaussian(link='log'), method='GCV')`, n=60: reported
  GCV 38.3 vs the accepted 4.43; mgcv reports 4.84 on the same data).
  `select_lambdas` now also tracks the best-scoring evaluation's mean and
  leads the branch-resolution candidates with it; the search path itself is
  unchanged, so selected smoothing parameters on unimodal problems are
  bit-identical. Whether the branch was lost depended on scipy's
  line-search trajectory, so this could surface as
  `tests/gam/test_sp_gradient.py::TestGradientFailureCases::`
  `test_multimodal_inner_fit_reports_search_branch` failing only on some
  scipy versions (e.g. 1.17).
