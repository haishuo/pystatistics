"""NumPy batched building blocks for the EM E-step.

Extracted from ``_em_batched.py`` on 2026-04-20 to keep each file under
the 500-SLOC hard limit (Coding Bible rule 4). See module ``_em_batched``
for a compatibility shim that re-exports these symbols.

All functions are FP64 and bit-faithful to the scalar reference
implementation on CPU.

The ``chi_square_mcar_batched_np`` function that lived here through
2.3.x was removed in 3.0.0 along with ``mom_mcar_test`` (which was its
only caller); the MCAR chi-square machinery now lives in Lacuna.
"""
from __future__ import annotations

import numpy as np

from pystatistics.mvnmle.backends._em_batched_patterns import _BatchedPatternIndex


def compute_conditional_parameters_np(
    mu: np.ndarray,
    sigma: np.ndarray,
    index: _BatchedPatternIndex,
) -> tuple:
    """Batched computation of per-pattern conditional regression matrices.

    For each pattern k with observed indices O_k and missing indices M_k,
    compute in a single batched Cholesky + batched solve:

        beta_k = Sigma[M_k, O_k] @ Sigma[O_k, O_k]^{-1}
        cond_cov_k = Sigma[M_k, M_k] - beta_k @ Sigma[O_k, M_k]

    The per-pattern results are padded to ``(P, v_mis_max, v_obs_max)``
    and ``(P, v_mis_max, v_mis_max)`` respectively; the caller must
    apply ``index.mis_mask`` / ``index.obs_mask`` when consuming slices.

    Uses NumPy's batched linalg (cholesky / solve_triangular support
    leading batch dimensions since NumPy 1.8 / 1.13). Runs on CPU.

    Parameters
    ----------
    mu : (v,) float64
    sigma : (v, v) float64
    index : _BatchedPatternIndex

    Returns
    -------
    beta_batched : (P, v_mis_max, v_obs_max)
    cond_cov_batched : (P, v_mis_max, v_mis_max)
    """
    P = index.n_patterns
    v_obs_max = index.v_obs_max
    v_mis_max = index.v_mis_max

    # Gather sigma_oo via 2D advanced indexing.
    # row_idx[k, i, j] = obs_idx[k, i]; col_idx[k, i, j] = obs_idx[k, j]
    row_idx = index.obs_idx[:, :, None]          # (P, v_obs_max, 1)
    col_idx = index.obs_idx[:, None, :]          # (P, 1, v_obs_max)
    sigma_oo = sigma[row_idx, col_idx]           # (P, v_obs_max, v_obs_max)

    # Replace padded rows/cols with identity so cholesky stays well-defined.
    mask_oo = index.obs_mask[:, :, None] & index.obs_mask[:, None, :]
    eye_oo = np.broadcast_to(
        np.eye(v_obs_max, dtype=sigma.dtype), sigma_oo.shape,
    )
    sigma_oo = np.where(mask_oo, sigma_oo, eye_oo)

    # Gather sigma_mo: rows are missing, cols are observed.
    mrow_idx = index.mis_idx[:, :, None]         # (P, v_mis_max, 1)
    sigma_mo = sigma[mrow_idx, col_idx]          # (P, v_mis_max, v_obs_max)
    # Zero out invalid rows/cols (missing side: mis_mask; observed side: obs_mask)
    valid_mo = index.mis_mask[:, :, None] & index.obs_mask[:, None, :]
    sigma_mo = np.where(valid_mo, sigma_mo, 0.0)

    # Gather sigma_mm.
    mcol_idx = index.mis_idx[:, None, :]         # (P, 1, v_mis_max)
    sigma_mm = sigma[mrow_idx, mcol_idx]         # (P, v_mis_max, v_mis_max)
    valid_mm = index.mis_mask[:, :, None] & index.mis_mask[:, None, :]
    sigma_mm = np.where(valid_mm, sigma_mm, 0.0)

    # (Note: an earlier revision computed a batched Cholesky here whose
    # factor was never used downstream. Removed — np.linalg.solve below
    # handles the per-pattern solve directly. Kept the pinv fallback
    # because np.linalg.solve still fails on strictly-singular sigma_oo,
    # which real tabular data with integer-encoded categoricals can
    # produce at the per-pattern sub-block level.)

    # Batched solve: beta^T = Sigma_oo^{-1} @ sigma_om = solve(sigma_oo, sigma_om)
    # sigma_om = sigma_mo^T  →  (P, v_obs_max, v_mis_max)
    sigma_om = np.swapaxes(sigma_mo, -1, -2)
    try:
        beta_T = np.linalg.solve(sigma_oo, sigma_om)
    except np.linalg.LinAlgError:
        # Per-pattern sigma_oo sub-block is singular. Fall back to pinv
        # for this batch. Issue a warning so the event is visible.
        import warnings
        warnings.warn(
            "e_step_batched_np: at least one per-pattern sigma_oo "
            "sub-block is numerically singular; falling back to "
            "Moore-Penrose pseudo-inverse for the batch.",
            UserWarning, stacklevel=3,
        )
        beta_T = np.matmul(np.linalg.pinv(sigma_oo), sigma_om)
    beta = np.swapaxes(beta_T, -1, -2)            # (P, v_mis_max, v_obs_max)

    # cond_cov = sigma_mm - beta @ sigma_om = sigma_mm - sigma_mo @ beta^T
    cond_cov = sigma_mm - np.matmul(beta, sigma_om)

    return beta, cond_cov


def compute_loglik_batched_np(
    mu: np.ndarray,
    sigma: np.ndarray,
    patterns,
    index: _BatchedPatternIndex,
) -> float:
    """Fully batched observed-data log-likelihood.

    Three batched operations replace the P-long Python loop of scalar
    cholesky/solve/slogdet that previously dominated SQUAREM wall-
    clock on many-pattern data:

      1. Batched Cholesky of (P, v_obs_max, v_obs_max) sigma_oo blocks
         (identity-padded in unused slots) → per-pattern log|Sigma_oo|.
      2. Observation-level gather: for each of N observations, pull
         the v_obs_max-vector of centred-data values at its pattern's
         observed column positions (with 0 at padded slots).
      3. One batched solve of (N, v_obs_max, v_obs_max) Sigma_oo
         blocks against the (N, v_obs_max) centred vectors, scattered
         via ``obs_pattern_id``; sum of squared residuals gives the
         quadratic-form contribution to log-likelihood.

    No per-pattern Python loop. Memory cost is (N, v_obs_max, v_obs_max)
    for the per-obs sigma-block gather, which for breast_cancer at
    N=569, v_obs_max=30 is ~4 MB — fine.
    """
    v_obs_max = index.v_obs_max

    # --- Log-determinant term (batched Cholesky over patterns) -----
    row_idx = index.obs_idx[:, :, None]
    col_idx = index.obs_idx[:, None, :]
    sigma_oo = sigma[row_idx, col_idx]
    mask_oo = index.obs_mask[:, :, None] & index.obs_mask[:, None, :]
    eye_oo = np.broadcast_to(np.eye(v_obs_max, dtype=sigma.dtype), sigma_oo.shape)
    sigma_oo = np.where(mask_oo, sigma_oo, eye_oo)

    # Batched Cholesky for log-det and the quadratic-form solve below.
    # Per-pattern sigma_oo sub-blocks can be numerically indefinite from
    # FP64 roundoff even when the global sigma is PD (confirmed on
    # credit_card_default via Project Lacuna). Apply a tiny diagonal
    # ridge before Cholesky rather than raising — the ridge (1e-12 on a
    # matrix normalised to trace ~v) is below any statistical precision.
    try:
        L_oo = np.linalg.cholesky(sigma_oo)
    except np.linalg.LinAlgError:
        import warnings
        ridge = 1e-10
        warnings.warn(
            f"e_step_full_batched_np: per-pattern sigma_oo indefinite; "
            f"retrying Cholesky with diagonal ridge {ridge:.0e}.",
            UserWarning, stacklevel=3,
        )
        eye_full = np.broadcast_to(
            np.eye(v_obs_max, dtype=sigma.dtype), sigma_oo.shape,
        )
        L_oo = np.linalg.cholesky(sigma_oo + ridge * eye_full)
    log_diag = np.log(np.diagonal(L_oo, axis1=-2, axis2=-1))
    logdet_per_pattern = 2.0 * np.sum(log_diag * index.obs_mask, axis=-1)

    # --- Quadratic-form term (fully batched over observations) -----
    # Per-observation gather of the pattern's observed indices. Padded
    # slots reuse obs_idx[0] harmlessly because the mask zeroes them.
    obs_pattern_id = index.obs_pattern_id
    per_obs_obs_idx = index.obs_idx[obs_pattern_id]            # (N, v_obs_max)
    per_obs_obs_mask = index.obs_mask[obs_pattern_id]          # (N, v_obs_max)

    # Gather each observation's data at its pattern's observed columns.
    # data_padded holds the real observed values at the right positions;
    # we pull them out in the pattern's canonical order.
    N_arange = np.arange(index.data_padded.shape[0])[:, None]   # (N, 1)
    y_gathered = index.data_padded[N_arange, per_obs_obs_idx]  # (N, v_obs_max)
    mu_gathered = mu[per_obs_obs_idx] * per_obs_obs_mask       # (N, v_obs_max)
    centered = (y_gathered - mu_gathered) * per_obs_obs_mask   # (N, v_obs_max)

    # Gather each observation's Sigma_oo block and Cholesky factor.
    L_per_obs = L_oo[obs_pattern_id]                            # (N, v_obs_max, v_obs_max)

    # Solve L_i z_i = centered_i for every observation in one call.
    z = np.linalg.solve(L_per_obs, centered[:, :, None]).squeeze(-1)  # (N, v_obs_max)

    quad_total = float(np.sum(z * z))

    # Weighted sum of log-determinants via n_per_pattern.
    logdet_sum = float(np.sum(
        index.n_per_pattern.astype(mu.dtype) * logdet_per_pattern
    ))

    return float(-0.5 * logdet_sum - 0.5 * quad_total)
