"""
Rank-deficiency (collinearity) detection for fitted MVN covariances.

A rank-deficient input — two or more (near-)collinear variables — has no
interior maximum-likelihood estimate: the multivariate-normal likelihood
increases without bound as the fitted covariance approaches singularity. In
that regime the optimizer's own convergence flag is unreliable (it may stall
and report either success or failure), so the *fitted* covariance must be
inspected directly. This module owns that single check.

The degeneracy signal is the smallest eigenvalue of the correlation matrix
implied by the fitted covariance. Using the correlation matrix rather than
the covariance directly makes the measure scale-invariant: it depends only
on the collinearity structure, not on the units of each variable. That lets
a single threshold separate genuine rank-deficiency from variables that are
merely full-rank but ill-conditioned (e.g. measured on very different
scales), which a raw condition number on the covariance cannot do.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import SingularMatrixError, ValidationError

# Smallest correlation-matrix eigenvalue treated as full-rank. Calibrated
# empirically: rank-deficient fits floor at ~3e-6 (an exactly duplicated or
# affine column), while legitimate ill-conditioned datasets stay at >= 4e-4.
# 1e-5 sits in that gap; for two variables it corresponds to |corr| ~ 0.99999.
DEFAULT_COLLINEARITY_TOL: float = 1e-5

# A column is treated as (near-)constant when its observed range is at the
# floating-point NOISE FLOOR for its magnitude — machine epsilon times the
# largest |value|, times this many ULP of slack for accumulated round-off. Zero
# variance means the marginal MLE is a degenerate point mass and no interior
# maximum-likelihood estimate exists; the scale-invariant correlation guard
# (below) cannot see it (it divides each variable by its own standard deviation),
# so it is caught here, at the input boundary (Rule 2).
#
# The test tracks fp RESOLUTION, not a fixed fraction of magnitude, so it does
# NOT misfire on a genuinely-varying column with a large OFFSET: e.g.
# ``1e8 + N(0, 1e-3)`` has range ~6e-3, thousands of ULP above the noise floor
# (~2e-8 = eps*1e8), and is full-rank. The old fixed 1e-10-of-magnitude rtol had
# a threshold of 1e-2 at magnitude 1e8 and wrongly rejected that column as
# constant even though its MLE demonstrably exists.
CONSTANT_COLUMN_NOISE_ULP: float = 4096.0


def correlation_min_eigenvalue(sigma: NDArray[np.floating]) -> float:
    """Smallest eigenvalue of the correlation matrix implied by ``sigma``.

    A value at or near zero indicates (near-)collinear variables. The
    measure is scale-invariant. A non-finite ``sigma`` or a non-positive
    variance on the diagonal is itself a degenerate fit and returns ``0.0``.

    Parameters
    ----------
    sigma : ndarray
        Square (p, p) fitted covariance matrix.

    Returns
    -------
    float
        The smallest eigenvalue of the derived correlation matrix, in the
        range [0, p]. ``0.0`` for an already-degenerate ``sigma``.
    """
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValidationError(
            f"sigma must be a square matrix, got shape {sigma.shape}"
        )
    if not np.all(np.isfinite(sigma)):
        return 0.0
    diag = np.diag(sigma)
    if np.any(diag <= 0.0):
        return 0.0
    d = np.sqrt(diag)
    corr = sigma / np.outer(d, d)
    eigvals = np.linalg.eigvalsh(corr)  # ascending order
    return float(eigvals[0])


def _constant_columns(data: NDArray[np.floating]) -> list[int]:
    """Indices of (near-)constant observed columns in an incomplete data matrix.

    A column is constant when the range of its *observed* (non-NaN) values is at
    the floating-point noise floor for its magnitude
    (:data:`CONSTANT_COLUMN_NOISE_ULP` ULP of the largest |value|) — so a column
    of identical values is flagged at any scale, while a genuinely-varying column
    is not, even when its spread is tiny relative to a large offset. A column with
    fewer than two observed values also carries no variance information and is
    reported.
    """
    if data.ndim != 2:
        raise ValidationError(f"data must be 2-D (n, p), got shape {data.shape}")
    eps = float(np.finfo(np.float64).eps)
    bad: list[int] = []
    for j in range(data.shape[1]):
        col = data[:, j]
        obs = col[~np.isnan(col)]
        if obs.size < 2:
            bad.append(j)
            continue
        scale = max(float(np.max(np.abs(obs))), 1.0)
        noise_floor = CONSTANT_COLUMN_NOISE_ULP * eps * scale
        if float(np.ptp(obs)) <= noise_floor:
            bad.append(j)
    return bad


def check_observed_variances(
    data: NDArray[np.floating],
    *,
    force: bool = False,
) -> str | None:
    """Guard the input against (near-)constant columns before fitting.

    A constant column has zero variance, so its marginal likelihood is a
    degenerate point mass: the MLE does not exist (the observed-data
    log-likelihood is unbounded as the fitted variance approaches zero), and a
    naive fit returns ``converged=True`` with a meaningless, arbitrarily large
    log-likelihood. Because this degeneracy is invisible to the scale-invariant
    fitted-covariance guard (which divides each variable by its own standard
    deviation), it is detected here, at the input boundary.

    Returns ``None`` when every column varies. When one or more columns are
    constant *and* ``force`` is True, returns a warning message (the caller marks
    the fit not-converged and attaches it). Otherwise raises
    :class:`SingularMatrixError`.
    """
    bad = _constant_columns(np.asarray(data))
    if not bad:
        return None

    detail = (
        f"column(s) {bad} have (near-)constant observed values (zero variance), "
        f"so no interior maximum-likelihood estimate exists — the observed-data "
        f"log-likelihood is unbounded as the fitted variance approaches zero and "
        f"the reported fit is not a true optimum"
    )
    if force:
        return (
            f"Degenerate fit accepted under force=True: {detail}. "
            f"Treat muhat and sigmahat with caution."
        )
    raise SingularMatrixError(
        f"MVN MLE failed: {detail}. Remove the constant column(s) before "
        f"fitting, or pass force=True to obtain the (non-converged) result anyway."
    )


def check_fitted_covariance(
    sigma: NDArray[np.floating],
    *,
    tol: float = DEFAULT_COLLINEARITY_TOL,
    force: bool = False,
) -> str | None:
    """Guard a fitted covariance against rank-deficiency.

    Parameters
    ----------
    sigma : ndarray
        Fitted covariance matrix to inspect.
    tol : float
        Full-rank threshold on the correlation-matrix minimum eigenvalue.
    force : bool
        When True, a degenerate fit is accepted rather than rejected: the
        function returns a warning message instead of raising.

    Returns
    -------
    str or None
        ``None`` when the fit is full-rank (no action needed). When the fit
        is rank-deficient *and* ``force`` is True, a warning message
        describing the degeneracy — the caller should attach it to the
        result and mark the fit not-converged.

    Raises
    ------
    SingularMatrixError
        When the fit is rank-deficient and ``force`` is False.
    """
    min_eig = correlation_min_eigenvalue(sigma)
    if min_eig >= tol:
        return None

    detail = (
        f"the fitted covariance is rank-deficient (correlation-matrix "
        f"minimum eigenvalue {min_eig:.2e} < {tol:.0e}). The input has "
        f"(near-)collinear variables, so no interior maximum-likelihood "
        f"estimate exists and the reported fit is not a true optimum"
    )
    if force:
        return (
            f"Degenerate fit accepted under force=True: {detail}. "
            f"Treat muhat and sigmahat with caution."
        )
    raise SingularMatrixError(
        f"MVN MLE failed: {detail}. Remove the collinear column(s) before "
        f"fitting, or pass force=True to obtain the (non-converged) result "
        f"anyway. The detection threshold is adjustable via collinearity_tol."
    )
