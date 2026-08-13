"""
``polyreg`` — multinomial logistic-regression imputation for unordered
categorical columns (R mice's default for unordered factors with >2 levels).

Fits a multinomial logit of the categorical target on the predictors, draws the
coefficient block once from its posterior normal approximation, predicts class
probabilities for the missing rows under the drawn coefficients, and samples a
category per row.

The fit is a **penalised Newton** with the same scale-aware slope ridge as the
``polr`` method (``lambda = 1e-5 * mean_column_second_moment * n_obs``, slopes
only — the per-class intercepts are unpenalised). The penalty enters the score
as well as the Hessian, so under the (quasi-)complete separation chained
equations routinely induce the fit shrinks to a finite optimum instead of
diverging. Before 6.1.x this method delegated to the unpenalised
``pystatistics.multinomial.multinom``: under complete separation that fit
"converged" at whatever saturated point L-BFGS-B stalled on, with a
near-singular information whose inverse blew up the posterior draw — the
imputed marginal could land anywhere (and the batched GPU counterpart of the
same unpenalised fit collapsed columns outright on fp32). The slope ridge is
the ``logreg``/``polr`` calibration applied to the multinomial member of the
family; because the intercepts are unpenalised, their score equations are exact
at the optimum and the in-sample predicted marginal equals the observed
marginal — the same identity the polr ridge preserves
(``tests/mice/test_polr_ridge_characterization.py``). On well-identified fits
the penalty is ~1e-5 relative to the information and the coefficients match R
``nnet::multinom`` to the R-validation tolerance.

The target arrives as consecutive ``0..K-1`` class indices and the method returns
indices; the chain handles the mapping to/from the column's category codes.
"""

from __future__ import annotations

import warnings

import numpy as np

from pystatistics.core.exceptions import (
    ConvergenceError,
    NumericalError,
    ValidationError,
)
from pystatistics.mice._encode import add_intercept
from pystatistics.mice.methods._draw import (
    marginal_indices,
    mvn_draw,
    sample_categories,
)
from pystatistics.mice.methods.registry import register

# Same constants as the sibling categorical methods (logreg/polr): relative
# ridge strength, Newton budget, and step tolerance.
_RIDGE = 1e-5
_MAX_NEWTON_ITER = 100
_NEWTON_TOL = 1e-8
# Backtracking budget for the damped Newton step: 30 halvings reach ~1e-9 of
# the full step, far past any useful progress on this convex objective.
_MAX_HALVINGS = 30


def _slope_ridge(X_obs: np.ndarray) -> float:
    """Scale-aware ridge coefficient for the multinomial slopes, mirroring
    ``methods/polr._slope_ridge`` exactly: ``lambda = _RIDGE *
    mean_column_second_moment * n_obs`` over the slope (non-intercept) columns.
    The mean column second moment makes it invariant to predictor scaling; the
    ``n_obs`` factor puts it on the scale of the n-scaled information so the
    relative penalty is ~``_RIDGE`` regardless of sample size."""
    n_obs = X_obs.shape[0]
    if n_obs == 0 or X_obs.shape[1] == 0:
        return 0.0
    diag_scale = float(np.mean(np.sum(X_obs * X_obs, axis=0))) / n_obs
    return _RIDGE * max(diag_scale, 1e-12) * n_obs


def _log_probs(eta_nr: np.ndarray) -> np.ndarray:
    """Full log-softmax with an appended zero column for the reference (last)
    class. ``eta_nr`` (n, K-1) -> log-probs (n, K)."""
    eta = np.hstack([eta_nr, np.zeros((eta_nr.shape[0], 1))])
    shift = eta - eta.max(axis=1, keepdims=True)
    return shift - np.log(np.exp(shift).sum(axis=1, keepdims=True))


def _penalised_hessian(Xa: np.ndarray, pnr: np.ndarray, pen: np.ndarray) -> np.ndarray:
    """Block Hessian of the penalised multinomial NLL, class-major layout
    matching ``coef.ravel()``: ``H[j,k] = X' diag(W_jk) X`` with
    ``W_jj = p_j(1-p_j)``, ``W_jk = -p_j p_k``, plus ``diag(pen)``."""
    n, P = Xa.shape
    knr = pnr.shape[1]
    d = knr * P
    H = np.empty((d, d), dtype=np.float64)
    for j in range(knr):
        pj = pnr[:, j]
        for k in range(j, knr):
            w = pj * (1.0 - pj) if j == k else -pj * pnr[:, k]
            block = (Xa * w[:, None]).T @ Xa
            H[j * P:(j + 1) * P, k * P:(k + 1) * P] = block
            if k != j:
                H[k * P:(k + 1) * P, j * P:(j + 1) * P] = block.T
    return H + np.diag(pen)


def _fit_multinomial(y: np.ndarray, Xa: np.ndarray, K: int):
    """Penalised Newton for the multinomial logit (last class = reference).

    ``y``: (n,) class indices 0..K-1; ``Xa``: (n, P) WITH intercept column 0.
    Returns ``(coef (K-1, P), cov (d, d))`` — the coefficient block and the
    posterior covariance ``(H + diag(pen))^{-1}`` at the optimum, ordered to
    match ``coef.ravel()``.

    Input contract (Rule 2): every class 0..K-1 present in ``y`` and
    ``n > (K-1) * P``; violations raise ``ValidationError``.
    """
    n, P = Xa.shape
    knr = K - 1
    d = knr * P

    counts = np.bincount(y, minlength=K)
    if np.any(counts == 0):
        raise ValidationError(
            f"polyreg target is missing observed classes: counts={counts.tolist()}"
        )
    if n <= d:
        # Over-parameterised sub-problem: n <= (K-1)*P means the unpenalised
        # MLE is not unique and the ridge-picked fit would be interpolation
        # with a penalty-dominated (meaningless) posterior draw. Refuse so the
        # caller's visible marginal-draw fallback handles it (same gate the
        # pre-6.1.x multinom delegation enforced).
        raise ValidationError(
            f"Insufficient observations: n={n} but model has {d} parameters. "
            f"Need n > (K-1)*P = {d}."
        )

    lam = _slope_ridge(Xa[:, 1:])
    pen = np.full(d, lam, dtype=np.float64)
    pen[0::P] = 0.0  # per-class intercepts unpenalised (polr precedent)

    y_onehot = np.zeros((n, K), dtype=np.float64)
    y_onehot[np.arange(n), y] = 1.0
    y_nr = y_onehot[:, :knr]

    # Start at the marginal log-odds intercepts (zero slopes) — the exact
    # analogue of logreg's beta0 start; a few Newton steps from the answer on
    # well-identified data.
    beta = np.zeros((knr, P), dtype=np.float64)
    beta[:, 0] = np.log(counts[:knr] / counts[K - 1])

    def pen_nll(b: np.ndarray) -> float:
        logp = _log_probs(Xa @ b.T)
        return float(-logp[np.arange(n), y].sum()
                     + 0.5 * lam * np.sum(b[:, 1:] * b[:, 1:]))

    f0 = pen_nll(beta)
    for _ in range(_MAX_NEWTON_ITER):
        pnr = np.exp(_log_probs(Xa @ beta.T))[:, :knr]
        grad = (Xa.T @ (y_nr - pnr)).T.ravel() - pen * beta.ravel()
        H = _penalised_hessian(Xa, pnr, pen)
        delta = np.linalg.solve(H, grad).reshape(knr, P)

        # Damped step: halve until the penalised NLL decreases (small relative
        # slack absorbs FP noise at the full step near convergence). Undamped
        # Newton can overshoot on saturated logits far from the optimum; the
        # objective is convex, so a decreasing step always exists.
        step = 1.0
        for _ in range(_MAX_HALVINGS):
            trial = beta + step * delta
            f1 = pen_nll(trial)
            if np.isfinite(f1) and f1 <= f0 + 1e-8 * (1.0 + abs(f0)):
                break
            step *= 0.5
        else:
            raise NumericalError(
                "polyreg Newton found no decreasing step (degenerate fit)"
            )
        beta, f0 = trial, f1
        if step * np.max(np.abs(delta)) < _NEWTON_TOL:
            break
    else:
        raise ConvergenceError(
            f"polyreg Newton did not converge in {_MAX_NEWTON_ITER} iterations",
            iterations=_MAX_NEWTON_ITER,
            threshold=_NEWTON_TOL,
        )

    pnr = np.exp(_log_probs(Xa @ beta.T))[:, :knr]
    cov = np.linalg.inv(_penalised_hessian(Xa, pnr, pen))
    if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(cov))):
        raise NumericalError("polyreg fit or covariance is non-finite")
    return beta, cov


class PolyregMethod:
    """Multinomial logistic imputation (conforms to ImputationMethod)."""

    name = "polyreg"
    target_kind = "categorical"

    def impute(
        self,
        y_obs: np.ndarray,
        X_obs: np.ndarray,
        X_mis: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        y = np.asarray(y_obs, dtype=np.intp)
        K = int(y.max()) + 1
        Xa_obs = add_intercept(X_obs)
        Xa_mis = add_intercept(X_mis)

        # A fit can still fail on an awkward intermediate sweep state: an
        # over-parameterised sub-problem (n_obs <= (K-1)*P, which a
        # high-cardinality column with many predictors readily triggers), a
        # missing observed class, or a genuinely degenerate Newton. All are
        # recoverable — fall back to a marginal draw (visibly) rather than
        # crashing the whole mice() run. Rule-1 documented exception: local,
        # not silent, retried next iteration on the fresh sweep state.
        try:
            coef, cov = _fit_multinomial(y, Xa_obs, K)
        except (
            ConvergenceError,
            NumericalError,
            ValidationError,
            np.linalg.LinAlgError,
        ) as exc:
            warnings.warn(
                f"polyreg fit failed ({type(exc).__name__}); using a "
                f"marginal draw for this sweep step.",
                UserWarning,
                stacklevel=2,
            )
            return marginal_indices(y, Xa_mis.shape[0], rng)

        # Draw the whole coefficient block from N(coef, cov); cov is ordered to
        # match coef.ravel() (class-major blocks of P).
        beta_star = mvn_draw(coef.ravel(), cov, rng).reshape(K - 1, -1)
        probs = np.exp(_log_probs(Xa_mis @ beta_star.T))  # (n_mis, K)
        return sample_categories(probs, rng)


register(PolyregMethod())
