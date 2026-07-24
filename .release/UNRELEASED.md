# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- `mvnmle` GPU errors on badly-scaled input now give the correct diagnosis and
  remedy. Columns differing enormously in variance made the fp32 fit collapse the
  small-scale column, after which the rank-deficiency guard inspected the
  corrupted *fitted* covariance and raised `SingularMatrixError` claiming the
  INPUT had "(near-)collinear variables" — false for full-rank data (the CPU path
  fits it exactly), with a remedy ("remove the collinear columns") that is both
  wrong and destructive. The GPU initial-parameter guard now screens the column
  variance ratio and raises `NumericalError` naming it a SCALING issue with the
  correct remedy (standardize/rescale, or use `backend='cpu'`). Also fixes the
  related inconsistency where the same well-posed family raised
  `SingularMatrixError` at one scale and `NumericalError` at another.

- `mice` no longer rejects a constant, fully-observed categorical predictor. The
  "needs >= 2 observed levels" rule applied to every categorical column, but it is
  only meaningful for a column that must be IMPUTED; a fully-observed single-level
  categorical is a harmless constant predictor (treatment dummies drop the
  reference level, so it contributes no columns) and is now accepted, mirroring
  how a constant fully-observed numeric predictor already was. The rule still
  applies to categorical columns with missing values.

- Fixed `mvnmle`'s fp32 GPU path silently accepting a badly-wrong fit on
  near-collinear data (R12 no-silent-wrong). The rank-deficiency guard inspects
  the *fitted* covariance, but on near-collinear input the fp32 optimizer stalls
  at a non-stationary point whose fp32 covariance still looks full-rank — so the
  guard passed and `mlest(..., backend='gpu')` returned `converged=True` with no
  warning on a fit ~1100 nats below the fp64 optimum, while the fp64 CPU path
  correctly raised `SingularMatrixError` on the same data. The GPU initial-
  parameter guard now also applies the scale-invariant collinearity screen the
  CPU uses (correlation-matrix minimum eigenvalue vs `DEFAULT_COLLINEARITY_TOL`),
  evaluated on the RAW pairwise sample covariance (newly stashed as
  `sample_cov_raw`, since the regularized starting covariance floors exactly the
  small eigenvalue that reveals collinearity). Near-collinear input is now
  refused on GPU as it is on CPU; well-conditioned fits are unaffected, keeping
  the fp32 accept-set a subset of the fp64 accept-set. Found by the 6.0.0
  adversarial red-team (mvnmle fp32-classifier-attack).

- Fixed `mice` GPU (fp32) silently collapsing a categorical/binary/ordinal
  imputed column onto level 0 on a degenerate fit. The GPU samplers correctly
  emit an all-NaN sentinel so a degenerate fit reaches the backend's end-of-sweep
  non-finite guard, but `_gpu_encode.indices_to_codes` cast the index straight to
  `long` — turning the NaN sentinel into index 0 and gathering `levels[0]`,
  silently erasing the signal before the guard could see it. `indices_to_codes`
  now propagates a non-finite index as NaN, so a degenerate GPU categorical fit
  raises `ValidationError` (fail-loud) instead of collapsing. Also corrected the
  `_gpu_polyreg` docstring, which claimed a marginal-draw fallback that does not
  exist (the GPU categorical path is fail-loud-via-guard by design). Found by the
  6.0.0 adversarial red-team (mice fp32-nosilent-collapse).

- Fixed `mice` `logreg` collapsing an imputed binary column onto one class under
  (quasi-)complete separation. The ridge stabiliser was added only to the IRLS
  Hessian (`X'WX`), not the gradient, so at convergence the fit solved the
  *unpenalised* score equation and `beta` diverged (e.g. `[-906, 471, -3]`),
  driving every imputed cell to class 0 (`frac1 == 0`, `sd == 0`). It is now a
  genuine penalty (present in the gradient too) toward a weakly-informative
  centre — the marginal-rate intercept with zero slopes — and n-scaled to the
  likelihood curvature, mirroring `polr`'s in-objective l2. Under separation the
  fit stays finite and reproduces the marginal instead of collapsing; on
  well-identified columns the penalty is negligible and the CPU-vs-R binary
  fidelity is unchanged (`|Δmean|` 0.007, W1 0.005 vs R `mice`). Found by the
  6.0.0 adversarial red-team (mice separation-collapse).

- Fixed `mvnmle` (`solver='reference'`, and any direct-backend fit) reporting a
  spurious, enormous log-likelihood with `converged=True` on scale-disparate
  data. The R-exact objective returns a large finite failure sentinel (`1e20`)
  when its Givens reconstruction cannot be evaluated at a point (e.g. columns
  differing by ~1e12 in scale); `extract_parameters` turned that into
  `loglik = -5e19` and the backend still reported `converged=True`. The objective
  sentinel is now named (`_OBJECTIVE_FAILURE_SENTINEL`), `extract_parameters`
  returns `NaN` when it is hit rather than `-sentinel/2`, and the direct backend
  surfaces a non-finite loglik with a "standardize/rescale the columns" warning
  and `converged=False`. Well-posed fits are unchanged (loglik still matches the
  default path / R). Found by the 6.0.0 adversarial red-team (mvnmle
  api-default-contract).

- Fixed `mvnmle`'s constant-column guard falsely rejecting a genuinely-varying
  column with a large offset. The zero-variance guard (`_constant_columns` in
  `mvnmle/_degeneracy.py`) flagged a column whose observed range was <= a fixed
  `1e-10 * magnitude`; for a column like `1e8 + N(0, 1e-3)` that threshold was
  `1e-2`, exceeding the genuine spread (~6e-3), so `mlest` raised
  `SingularMatrixError` ("no interior MLE exists") on data whose MLE demonstrably
  exists (`force=True` recovered the variance exactly). The test now uses a
  machine-epsilon noise floor (`CONSTANT_COLUMN_NOISE_ULP * eps * magnitude`) so
  it tracks floating-point resolution: genuinely-varying columns fit regardless
  of offset, while truly-constant columns are still rejected at any scale (the
  M1 guard is preserved). Found by the 6.0.0 adversarial red-team (mvnmle
  conditioning-boundary).

- Fixed `mice` crashing the entire run on an unfittable categorical
  sub-problem. `polyreg`/`polr` fall back to a marginal draw when a fit fails,
  but the fallback caught only `ConvergenceError`/`NumericalError`/`LinAlgError`
  — not the `ValidationError` that `multinom` raises when the model is
  over-parameterized (`n_obs <= (n_classes-1)*n_params`), which a
  high-cardinality categorical column with many predictors readily triggers.
  That exception propagated out of the sweep and aborted `mice()`. `polyreg`
  and `polr` now include `ValidationError` in the caught set, so the step
  degrades to a (warned) marginal draw and the run completes. Found by the
  6.0.0 adversarial red-team (mice categorical-edges).

- `mice` now warns when a numeric column that needs imputation has
  (near-)constant observed values. Such a column gave pmm/norm no variance to
  model, so every imputed cell equalled the constant with zero uncertainty —
  silently collapsing the between-imputation variance and the fraction of
  missing information to 0 in Rubin's rules. It now emits a `UserWarning`
  (`_warn_constant_numeric_columns` in `mice/design.py`), mirroring R `mice`'s
  logged 'constant' event. Categorical constants were already rejected by level
  resolution. Found by the 6.0.0 adversarial red-team (mice missingness
  pathology).

- Fixed `mice.pool` (Rubin's rules) on degenerate zero within-imputation
  variance (`ubar == 0` with positive between-imputation variance). The old
  code derived `riv` from `ubar` but `lambda_` from `total` via separate
  guards, so in this case they silently disagreed — `riv` was reported as `0`
  while `lambda_` was `1` (mathematically impossible, since
  `lambda = riv/(1+riv)`), `fmi` was wrong, and the confidence interval went
  silently `NaN`. `riv` is now derived from `lambda_` so the two are always
  consistent (`riv = +inf`, `lambda_ = 1`, `fmi = 1` in the degenerate case),
  and `pool` now emits a `UserWarning` explaining that within-imputation
  variance is zero and the interval is undefined, instead of returning a
  silent `NaN`. Normal-case pooled numbers are unchanged. Found by the 6.0.0
  adversarial red-team (mice Rubin-edges).

- Docs: fixed malformed reStructuredText in numerous docstrings so the
  Sphinx API reference renders cleanly. Example blocks, algorithm
  listings, equations, and option lists in ``pca``, ``boot``, ``multinom``,
  ``mlest``, ``mlest_monotone_closed_form``, ``adf_test``, ``pacf``,
  ``Gaussian.aic``, ``HTestSolution.summary`` and the ``timeseries``
  package overview are now proper literal blocks and lists instead of
  being silently dropped or misindented. Also removed duplicate API
  entries (``DataSource``, time-series result attributes) and ambiguous
  cross-references from shape annotations. No behavior, signature, or
  documented-value changes — docstring formatting only. The docs now
  build warning-free under ``sphinx-build -W``.
