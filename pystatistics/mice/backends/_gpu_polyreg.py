"""
Batched multinomial-logit imputation on the GPU (MICE ``polyreg``).

GPU counterpart of ``methods/polyreg.py``, vectorized over the ``m`` imputation
chains. R mice's default for unordered factors with >2 levels: fit a multinomial
logit of the K-level target on the predictors, draw the coefficient block once
from its posterior normal approximation, predict class probabilities for the
missing rows, and sample a category per row.

Model (identical to the CPU path): softmax with the LAST class as the reference
(``beta_ref = 0``), coefficients ``(K-1, P)``, fitted by **penalised batched
Newton** with the CPU method's scale-aware slope ridge ported verbatim:
``lambda = 1e-5 * mean_slope_column_second_moment * n_obs`` on the slopes, the
per-class intercepts unpenalised. The penalty enters the gradient
(``- pen * beta``) as well as the Hessian — a Hessian-only ridge (the pre-6.1.x
form this file carried, the same defect as the GSS logreg refusal) leaves the
score equation UNPENALISED, so under the (quasi-)complete separation chained
equations routinely induce beta still diverged: on fp32-only MPS all the way to
a saturated ~1e26 fit whose posterior draw collapsed every missing cell onto
ONE category — silent corruption the end-of-sweep guard cannot see, because the
collapse is finite. With the penalty, a separated fit shrinks to the same
finite optimum as the CPU reference.

Each step forms the penalised multinomial block Hessian

    H[j, k] = X' diag(W_jk) X + diag(pen),  W_jj = p_j(1-p_j), W_jk = -p_j p_k

over the ``K-1`` non-reference classes (class-major block layout matching
``coef.ravel()``), and solves ``H delta = grad`` via ``cholesky_ex`` + the shared
device-split SPD apply (matmul-series inverse on MPS). Each step is damped by a
per-chain halving line search on the penalised NLL — the CPU method's damping
ported verbatim. This is load-bearing, exactly as in the polr Newton: the
UNPENALISED intercept block loses its curvature when the logits saturate (W ->
0 with no ridge underneath), so the full Newton step overshoots by orders of
magnitude under separation and the iterate diverges even though the penalised
optimum is finite (measured: undamped fp64 beta ~2e10 vs the true optimum ~15
on a completely separated recode). Per-chain convergence freezing on the DAMPED
step; bounded iterations; a non-finite fp32 step is zeroed (the polr Newton
guard) so a transient bad solve freezes the chain at its last finite iterate
instead of poisoning the whole sweep step. The step tolerance is floored at
1e-5 in FP32 — the fp64 tolerance of 1e-8 sits BELOW the fp32 solve noise floor
(measured ~2e-8-4e-8 stall on MPS), so without the floor every fp32 fit burned
the full iteration budget without ever reporting convergence (the same
precision reasoning as the multinom solver's documented fp32 tol floor).

Posterior draw ``beta* ~ N(beta_hat, H^{-1})`` with ``H`` the penalised Hessian
at the optimum — the CPU ``cov = (H + diag(pen))^{-1}`` exactly — then softmax
prediction and inverse-CDF category sampling, the tensor counterparts of
``_draw.mvn_draw`` and ``_draw.sample_categories``.

Within one sweep step every chain shares the same target column, so ``K`` is
fixed across the batch. The one visible fallback mirrors the CPU method's
over-parameterisation gate: ``n_obs <= (K-1)*P`` has no unique MLE and a
penalty-dominated draw, so the whole step falls back to a batched marginal draw
with the same UserWarning as the CPU fallback (the condition is static in the
shapes — no device sync). A genuinely degenerate fit otherwise surfaces as a
non-finite (NaN) imputation that the backend's end-of-sweep guard catches and
raises on — fail-loud, not silent. That sentinel must be preserved end-to-end:
``_gpu_encode.indices_to_codes`` propagates a non-finite index as NaN rather
than casting it to level 0 (the defect fixed in 6.0.x). All randomness flows
through the passed generator (Rule 6). Matches the CPU reference
distributionally at the GPU/FP32 tolerance tier.
"""

from __future__ import annotations

import warnings

from pystatistics.core.exceptions import ValidationError

from pystatistics.mice.backends._gpu_linreg import (
    add_intercept,
    discrete_glm_compute_dtype,
    _cholesky_ridged,
)
from pystatistics.mice.backends._gpu_spd import apply_inv_factor_T, solve_spd

# Mirrors the CPU methods/polyreg.py constants: relative slope-ridge strength,
# Newton budget, step tolerance, and line-search halving budget.
_RIDGE = 1e-5
_MAX_NEWTON_ITER = 100
_NEWTON_TOL = 1e-8
# FP32 floor for the step tolerance (see module docstring).
_NEWTON_TOL_FP32 = 1e-5
_MAX_HALVINGS = 30


def _log_probs(eta_nonref):
    """Full log-softmax with an appended zero column for the reference (last)
    class. ``eta_nonref`` (m, n, K-1) -> log-probs (m, n, K)."""
    import torch

    m, n, _ = eta_nonref.shape
    zeros = torch.zeros((m, n, 1), dtype=eta_nonref.dtype, device=eta_nonref.device)
    eta = torch.cat([eta_nonref, zeros], dim=2)
    return eta - torch.logsumexp(eta, dim=2, keepdim=True)


def _penalty_diag(Xa, knr):
    """Per-chain penalty diagonal (m, (K-1)P): the CPU ``_slope_ridge`` formula
    ``lambda = _RIDGE * mean_slope_column_second_moment * n`` on the slope
    coordinates, zero on each class's intercept coordinate (column 0 of every
    class block — the polr precedent: unpenalised intercepts keep their score
    equations exact, so the in-sample predicted marginal equals the observed
    marginal even under separation)."""
    import torch

    m, n, P = Xa.shape
    d = knr * P
    diag_scale = (Xa[:, :, 1:] * Xa[:, :, 1:]).sum(dim=1).mean(dim=1) / max(n, 1)
    lam = _RIDGE * diag_scale.clamp_min(1e-12) * max(n, 1)           # (m,)
    pen = lam[:, None].repeat(1, d)                                  # (m, d)
    pen[:, 0::P] = 0.0
    return pen


def _multinomial_hessian(Xa, pnr, pen):
    """Penalised block Hessian of the multinomial NLL. ``Xa`` (m, n, P), ``pnr``
    (m, n, K-1) non-reference probabilities, ``pen`` (m, (K-1)P). Returns
    (m, (K-1)P, (K-1)P) in the class-major block layout matching
    ``coef.ravel()``."""
    import torch

    m, n, P = Xa.shape
    knr = pnr.shape[2]
    d = knr * P
    H = torch.zeros((m, d, d), dtype=Xa.dtype, device=Xa.device)
    for j in range(knr):
        pj = pnr[:, :, j]
        for k in range(j, knr):
            w = pj * (1.0 - pj) if j == k else -pj * pnr[:, :, k]
            block = (Xa * w.unsqueeze(-1)).transpose(1, 2) @ Xa       # (m, P, P)
            H[:, j * P:(j + 1) * P, k * P:(k + 1) * P] = block
            if k != j:
                H[:, k * P:(k + 1) * P, j * P:(j + 1) * P] = block.transpose(1, 2)
    return H + torch.diag_embed(pen)


def _pen_nll(beta, y_onehot, Xa, pen):
    """Per-chain penalised NLL, shape (m,). ``beta`` (m, K-1, P). The single
    source of truth for the line-search objective: ``-sum log p_y + 0.5 *
    sum(pen * beta^2)`` (``pen`` is already zero on the intercept coords)."""
    import torch

    m = beta.shape[0]
    logp = _log_probs(Xa @ beta.transpose(1, 2))                    # (m, n, K)
    fit = -(y_onehot * logp).sum(dim=(1, 2))
    b = beta.reshape(m, -1)
    return fit + 0.5 * (pen * b * b).sum(dim=1)


def _halving_step(beta, delta, f0, y_onehot, Xa, pen, frozen):
    """Per-chain halving line-search scale for the Newton step ``delta``
    (m, K-1, P) — the CPU method's damping, batched: halve from 1 until the
    penalised NLL decreases for that chain (small relative slack absorbs FP
    noise at the full step near convergence; the trial must also be FINITE, as
    in the polr search — ``NaN <= f0`` is always False, so a NaN-blind search
    would shrink forever and then commit a poisoned step). ``frozen`` chains
    are not searched. A chain that exhausts the budget without a decreasing
    step (unreachable for this convex objective short of genuine numerical
    degeneracy) gets step 0 with a NaN beta poison applied by the caller —
    fail loud via the end-of-sweep guard, mirroring the CPU NumericalError
    (which the CPU fallback converts to a visible marginal draw).

    Returns ``(step (m,), accepted (m,), f1 (m,))`` — ``f1`` is each accepted
    chain's objective at its accepted step, so the caller reuses it as the next
    iteration's ``f0`` instead of recomputing the NLL (the polr line-search
    optimisation). Evaluated without autograd."""
    import torch

    with torch.no_grad():
        slack = 1e-8 * (1.0 + f0.abs())
        step = torch.ones_like(f0)
        accepted = frozen.clone()
        for _ in range(_MAX_HALVINGS):
            trial = beta + step[:, None, None] * delta
            f1 = _pen_nll(trial, y_onehot, Xa, pen)
            ok = accepted | (torch.isfinite(f1) & (f1 <= f0 + slack))
            if bool(ok.all()):
                accepted = ok
                break
            step = torch.where(ok, step, 0.5 * step)
            accepted = ok
    return step, accepted, f1


def batched_multinomial_newton(y_onehot, Xa, n_classes):
    """Batched damped penalised Newton on the multinomial NLL (see module
    docstring). ``y_onehot`` (m, n, K), ``Xa`` (m, n, P) WITH intercept. Returns
    ``(beta_hat (m, K-1, P), L)`` — the coefficient block and the Cholesky of
    the penalised Hessian at ``beta_hat`` (for the posterior covariance
    ``H^{-1}``)."""
    import torch

    m, n, P = Xa.shape
    knr = n_classes - 1
    d = knr * P
    dtype, device = Xa.dtype, Xa.device

    pen = _penalty_diag(Xa, knr)                                     # (m, d)
    y_nr = y_onehot[:, :, :knr]

    # Start at the marginal log-odds intercepts, zero slopes — the CPU start.
    # Every class is present among the observed rows by construction of the
    # design's levels, so the counts are strictly positive; a violated contract
    # yields a non-finite start that the end-of-sweep guard raises on (Rule 1).
    counts = y_onehot.sum(dim=1)                                     # (m, K)
    beta = torch.zeros((m, knr, P), dtype=dtype, device=device)
    beta[:, :, 0] = torch.log(counts[:, :knr] / counts[:, knr:])

    # FP32 cannot reach the fp64 step tolerance (solve noise stalls the step at
    # ~2e-8-4e-8 > 1e-8, burning the full budget); floor it at 1e-5, far below
    # any statistically material coefficient change.
    tol = _NEWTON_TOL if dtype == torch.float64 else max(_NEWTON_TOL, _NEWTON_TOL_FP32)

    f0 = _pen_nll(beta, y_onehot, Xa, pen)                          # (m,)
    converged = torch.zeros(m, dtype=torch.bool, device=device)
    for _ in range(_MAX_NEWTON_ITER):
        eta_nr = Xa @ beta.transpose(1, 2)                          # (m, n, K-1)
        pnr = _log_probs(eta_nr).exp()[:, :, :knr]
        resid = y_nr - pnr                                          # (m, n, K-1)
        # Penalised score, class-major: the ridge enters the gradient, not just
        # the Hessian — at convergence beta solves the PENALISED equations, so
        # a separated fit shrinks to the finite CPU optimum instead of diverging.
        grad = (
            (Xa.transpose(1, 2) @ resid).transpose(1, 2).reshape(m, d)
            - pen * beta.reshape(m, d)
        ).unsqueeze(-1)                                             # (m, d, 1)
        L = _cholesky_ridged(_multinomial_hessian(Xa, pnr, pen))
        delta = solve_spd(L, grad).squeeze(-1)                      # (m, d)
        # An fp32 device solve can transiently return a non-finite step (as in
        # the polr/logreg Newton). Zero it so it can never poison ``beta`` —
        # the chain freezes at its last finite iterate via ``small`` below
        # instead of emitting an all-NaN fit for the sweep step.
        delta = torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
        delta = delta.reshape(m, knr, P)

        active = ~converged
        step, accepted, f1 = _halving_step(
            beta, delta, f0, y_onehot, Xa, pen, converged
        )
        # Poison chains whose line search found no decreasing step (the CPU
        # raises NumericalError here): NaN beta -> NaN imputation -> guard.
        beta = torch.where(
            accepted[:, None, None], beta, torch.full_like(beta, float("nan"))
        )
        apply = active.to(dtype) * step                             # frozen -> 0
        update = apply[:, None, None] * delta
        small = update.abs().amax(dim=(1, 2)) < tol
        beta = beta + update
        converged = converged | (active & small)
        # f1 is the accepted-step objective for the chains that moved; frozen
        # chains keep their (still valid) f0.
        f0 = torch.where(active & accepted, f1, f0)
        if bool(converged.all()):
            break

    eta_nr = Xa @ beta.transpose(1, 2)
    pnr = _log_probs(eta_nr).exp()[:, :, :knr]
    L = _cholesky_ridged(_multinomial_hessian(Xa, pnr, pen))
    return beta, L


def _sample_categories(probs, gen):
    """Inverse-CDF sample one class per row from a (m, n, K) probability tensor.
    Robust to tiny negatives from an indefinite draw (clip + renormalise), the
    batched counterpart of ``_draw.sample_categories``. Returns (m, n) of class
    indices as a float tensor.

    Fail loud (Rule 1): a row with any non-finite probability yields ``NaN``,
    never a silent category 0 (under NaN, ``(cdf >= u)`` is all-False and
    ``argmax`` returns 0). Emitting NaN lets a degenerate fit reach the backend's
    end-of-sweep non-finite guard. Finite rows are unaffected."""
    import torch

    finite_row = torch.isfinite(probs).all(dim=2)                  # (m, n)
    safe = torch.where(finite_row.unsqueeze(2), probs, torch.zeros_like(probs))
    safe = safe.clamp_min(0.0)
    safe = safe / safe.sum(dim=2, keepdim=True).clamp_min(torch.finfo(safe.dtype).tiny)
    cdf = safe.cumsum(dim=2)
    u = torch.rand(
        safe.shape[:2] + (1,), generator=gen, dtype=safe.dtype, device=safe.device
    )
    idx = (cdf >= u).to(torch.int64).argmax(dim=2).to(safe.dtype)
    return torch.where(finite_row, idx, torch.full_like(idx, float("nan")))


def gpu_polyreg_impute(y_obs, X_obs, X_mis, gen, *, donors=None, n_classes=None):
    """Batched penalised multinomial-logit imputation (R ``polyreg``).

    ``y_obs`` (m, n_obs) of 0..K-1 class indices, ``X_obs`` (m, n_obs, q),
    ``X_mis`` (m, n_mis, q); ``n_classes`` = K is required (passed by the sweep
    from the column's level count). Returns (m, n_mis) of 0..K-1 indices.
    ``donors`` is accepted and ignored (uniform method signature).
    """
    import torch

    if n_classes is None:
        raise ValidationError("gpu_polyreg_impute requires n_classes (number of levels)")
    K = int(n_classes)

    out_dtype = X_obs.dtype
    compute_dtype = discrete_glm_compute_dtype(X_obs.device, out_dtype)
    X_obs = X_obs.to(compute_dtype)
    X_mis = X_mis.to(compute_dtype)

    Xa = add_intercept(X_obs)
    m, n, P = Xa.shape
    knr = K - 1
    d = knr * P
    n_mis = X_mis.shape[1]
    y_onehot = torch.zeros((m, n, K), dtype=Xa.dtype, device=Xa.device)
    y_onehot.scatter_(2, y_obs.to(torch.int64).unsqueeze(-1), 1.0)

    if n <= d:
        # The CPU method's over-parameterisation gate, ported: no unique MLE,
        # penalty-dominated draw. Fall back to a batched marginal draw for the
        # whole step — visibly, with the CPU fallback's warning. The condition
        # is static in the shapes, so this branch is deterministic and costs no
        # device sync; the next sweep retries the full model.
        warnings.warn(
            f"polyreg fit failed (ValidationError: n_obs={n} <= (K-1)*P={d}); "
            f"using a marginal draw for this sweep step.",
            UserWarning,
            stacklevel=2,
        )
        p_marg = (y_onehot.sum(dim=1) / max(n, 1)).unsqueeze(1)      # (m, 1, K)
        probs = p_marg.expand(m, n_mis, K)
        return _sample_categories(probs, gen).to(out_dtype)

    beta_hat, L = batched_multinomial_newton(y_onehot, Xa, K)
    z = torch.randn((m, d), generator=gen, dtype=Xa.dtype, device=Xa.device)
    beta_star = beta_hat.reshape(m, d) + apply_inv_factor_T(L, z.unsqueeze(-1)).squeeze(-1)
    beta_star = beta_star.reshape(m, knr, P)

    Xa_mis = add_intercept(X_mis)
    eta_nr = Xa_mis @ beta_star.transpose(1, 2)
    probs = _log_probs(eta_nr).exp()
    return _sample_categories(probs, gen).to(out_dtype)
