"""
Batched Bayesian logistic-regression imputation on the GPU (MICE ``logreg``).

GPU counterpart of ``methods/logreg.py``, vectorized over the ``m`` imputation
chains as the leading batch dimension. At a sweep step every chain fits the
*same* binary target on the *same* observed rows but with different predictor
values (each chain imputed the other columns differently), so we run ``m``
independent logistic fits at once with batched kernels.

Model (identical to the CPU path, R mice's default for two-level factors):

  * penalised IRLS (Newton on the concave penalised logistic log-likelihood).
    The penalty is the CPU ``methods/logreg.py`` one, ported verbatim: a GENUINE
    ridge toward a weakly-informative centre ``beta0 = [logit(marginal rate),
    0, ...]``, entering the gradient (``- R (beta - beta0)``) as well as the
    Gram ``G = X'WX + R``. A Hessian-only ridge (the pre-6.0.1 form this file
    carried until 6.1.x) leaves the score equation UNPENALISED, so under the
    (quasi-)complete separation chained equations routinely induce beta still
    diverged — on fp32-only MPS all the way to a non-finite fit, whose NaN
    sentinel had the end-of-sweep guard refuse whole real-survey runs (GSS).
    With the penalty, a separated fit shrinks toward the marginal rate and
    stays finite, exactly like the CPU reference.
    Each step solves ``G·delta = X'(y - p) - R(beta - beta0)`` through
    ``cholesky_ex`` plus the matmul-series triangular inverse on MPS
    (``solve_triangular`` is ~250x slower there) and ``solve_triangular`` on
    CUDA/CPU — the shared device split in ``core.compute.linalg``. Iteration is
    bounded (cap 50) with per-chain convergence freezing: a chain stops
    updating once ``max|delta| < tol``. The step tolerance is floored at 1e-5
    in FP32 — the fp64 tolerance of 1e-8 sits BELOW the fp32 solve noise floor,
    which is data-dependent (measured on MPS: ~3e-8-4e-8 on a well-identified
    n=1000 fixture, up to ~4e-5 on GSS n=10,000 survey columns), so without the
    floor NO fp32 fit ever reported convergence and every call burned the full
    50-iteration budget (same precision reasoning as the polyreg Newton and the
    multinom solver's documented fp32 tol floor). With it a well-identified fit
    converges in ~4-6 iterations.
    Each step is additionally globalised by a per-chain halving line search on
    the penalised NLL (``_halving_step``, the polyreg/polr pattern): full
    steps are accepted whenever they descend, so well-identified fits are
    unchanged, while separated/ill-conditioned fits can no longer oscillate.
    Without it the undamped steps grew ~13 -> 2e5 on real-survey designs whose
    structurally-zero dummy columns dilute the mean-moment ridge (penalised
    optimum at |beta| ~ 1e4), and in fp32 a transient overshoot reached inf
    and froze the chain at a non-finite iterate whose posterior draw was
    all-NaN — the GSS n=50k MPS refusal that survived the penalty alone. A
    chain with no observed-descent step within the halving budget freezes at
    its last finite iterate (evaluation noise near a separated optimum, not
    divergence — the polr semantics); it can never freeze non-finite, because
    every accepted step descended from a finite start.
  * posterior draw ``beta* ~ N(beta_hat, G^{-1})`` — the normal approximation
    around the (penalised) MLE that R's ``logreg`` uses, with ``G`` the ridged
    Gram, matching the CPU ``cov = (X'WX + R)^{-1}``. With ``G = L L'`` the
    draw is ``beta* = beta_hat + L^{-T} z`` (``Var = L^{-T} L^{-1} = G^{-1}``).
    Logistic has no dispersion parameter, so unlike the Gaussian draw there is
    no sigma / chi-square term.
  * predict ``p = sigmoid(X_mis beta*)`` and sample 0/1 by ``u < p``.

The target arrives as 0/1 class indices (the sweep maps the column's two category
codes to indices) and this returns 0/1 indices, batched (m, n_mis).

Like the CPU method there is no marginal-draw fallback: the penalty plus the
linear-predictor clip keep the fit finite even under quasi-/complete separation
(where the predicted probabilities saturate cleanly), and genuine degeneracy is
caught by the backend's end-of-sweep non-finite guard. All randomness flows
through the passed generator (Rule 6). Results match the CPU reference
distributionally at the GPU/FP32 tolerance tier.
"""

from __future__ import annotations

from pystatistics.mice.backends._gpu_linreg import (
    add_intercept,
    discrete_glm_compute_dtype,
)
from pystatistics.mice.backends._gpu_spd import apply_inv_factor_T, solve_spd

# Mirrors the CPU logreg constants (methods/logreg.py): relative ridge on X'WX
# (also tames separation), Newton cap and tolerance, and the linear-predictor
# clip that keeps exp() from overflowing and weights strictly positive.
_RIDGE = 1e-5
_MAX_IRLS_ITER = 50
_IRLS_TOL = 1e-8
# FP32 floor for the step tolerance (see module docstring / polyreg precedent).
_IRLS_TOL_FP32 = 1e-5
_ETA_CLIP = 30.0
_W_FLOOR = 1e-9
# Line-search halving budget (the polyreg/polr globalisation, see
# ``_halving_step``): 30 halvings reach a step of ~1e-9, well past any useful
# progress, so a chain still not decreasing there is genuinely degenerate.
_MAX_HALVINGS = 30


def _ridged_gram(Xa, w, ridge_diag):
    """Reweighted penalised Gram ``X'WX + R`` per chain. ``w``: (m, n);
    ``ridge_diag``: (m, k) per-parameter penalty diagonal (R is diagonal)."""
    import torch

    XtWX = Xa.transpose(1, 2) @ (w.unsqueeze(-1) * Xa)        # (m, k, k)
    return XtWX + torch.diag_embed(ridge_diag)


def _pd_gram_cholesky(G):
    """Lower Cholesky of the penalised Gram with per-chain Levenberg-Marquardt
    escalation — the polr ``_pd_cholesky``, reused verbatim.

    The single tiny jitter ``_cholesky_ridged`` carried before 6.1.3 is
    calibrated to well-conditioned Grams. At survey n the fp32 Gram's accumulation error
    (~entries * eps ~ 5e-3 at n=50k, entries ~4e4) can EXCEED the penalty
    floor when structurally-zero dummy columns dilute the mean-moment ridge
    (~0.03) — numerical PD-ness then depends on each chain's imputed values,
    and ``cholesky_ex`` fails for individual chains (measured: one chain of
    20 on GSS n=50k, every sweep, with a perfectly healthy |beta| ~ 1e0 fit —
    the factor, not the fit, was the last remaining NaN source). Escalating a
    tiny jitter x10 per try factors such chains at mu ~ 1e-2-1e-1 (a
    negligible widening of the posterior draw); well-conditioned chains
    factor on the first try, bit-identically to ``_cholesky_ridged``. A chain
    that never factors keeps its non-finite factor for the end-of-sweep
    guard."""
    import torch

    from pystatistics.mice.backends._gpu_polr import _pd_cholesky

    m, k = G.shape[0], G.shape[1]
    eye = torch.eye(k, dtype=G.dtype, device=G.device)
    zero = torch.zeros(m, dtype=G.dtype, device=G.device)
    return _pd_cholesky(G, zero, eye)


def _penalty_terms(y, Xa):
    """Per-chain penalty diagonal ``R`` (m, k) and centre ``beta0`` (m, k),
    mirroring the CPU ``methods/logreg._fit_logistic`` exactly.

    Slope penalty ``lam = _RIDGE * mean_col_second_moment * n`` — ~``_RIDGE``
    relative to the n-scaled curvature X'WX, so it is negligible on
    well-identified fits and only binds where the unpenalised MLE diverges
    ((quasi-)complete separation). The intercept is anchored toward the
    marginal logit at the small fixed strength ``_RIDGE * n`` so the baseline
    rate is preserved, not shrunk to 0.5."""
    import torch

    m, n, k = Xa.shape
    diag_scale = (Xa * Xa).sum(dim=1).mean(dim=1) / max(n, 1)        # (m,)
    lam = _RIDGE * diag_scale.clamp_min(1e-12) * max(n, 1)           # (m,)
    ridge_diag = lam[:, None].repeat(1, k)                           # (m, k)
    ridge_diag[:, 0] = _RIDGE * max(n, 1)     # anchor intercept to marginal logit

    pbar = y.mean(dim=1).clamp(1.0 / (n + 2.0), 1.0 - 1.0 / (n + 2.0))  # (m,)
    beta0 = torch.zeros((m, k), dtype=Xa.dtype, device=Xa.device)
    beta0[:, 0] = torch.log(pbar / (1.0 - pbar))
    return ridge_diag, beta0


def _pen_nll(beta, y, Xa, ridge_diag, beta0):
    """Per-chain penalised logistic NLL, shape (m,). The single source of truth
    for the line-search objective: ``-sum [y·log p + (1-y)·log(1-p)] + 0.5 *
    sum(R·(beta-beta0)^2)``, evaluated via ``logsigmoid`` on the RAW (unclamped)
    linear predictor — stable at any finite eta (``logsigmoid(-|eta|) ~ -|eta|``,
    no exp overflow), strictly convex and coercive in beta (every coordinate is
    penalised), so a decreasing step always exists and the accepted iterates
    stay bounded. The iteration itself keeps the eta clip (CPU parity); beyond
    the clip the two objectives differ by < 1e-13, absorbed by the slack."""
    import torch

    eta = (Xa @ beta.unsqueeze(-1)).squeeze(-1)                     # (m, n)
    fit = -(
        y * torch.nn.functional.logsigmoid(eta)
        + (1.0 - y) * torch.nn.functional.logsigmoid(-eta)
    ).sum(dim=1)
    dev = beta - beta0
    return fit + 0.5 * (ridge_diag * dev * dev).sum(dim=1)


def _halving_step(beta, delta, f0, y, Xa, ridge_diag, beta0, frozen):
    """Per-chain halving line-search scale for the IRLS step ``delta`` (m, k) —
    the polyreg/polr globalisation: halve from 1 until the penalised NLL
    decreases for that chain (a small relative slack absorbs FP noise at the
    full step near convergence; the trial must also be FINITE — ``NaN <= f0``
    is always False, so a NaN-blind search would shrink forever and then commit
    a poisoned step). ``frozen`` chains are not searched. A chain that exhausts
    the budget without an observed-descent step is reported unaccepted; the
    caller takes NO step for it and freezes it at its last finite iterate (the
    polr semantics — near a separated optimum the residual "increase" is
    evaluation noise, not divergence).

    This is load-bearing beyond (quasi-)separation: on real-survey designs with
    many structurally-zero dummy columns the mean-second-moment ridge is
    diluted and the penalised optimum sits at |beta| ~ 1e4; the undamped IRLS
    then OSCILLATES (measured step growth 13 -> 2e5 on GSS n=50k), and in fp32
    a transient overshoot reaches inf, freezing the chain at a non-finite
    iterate whose posterior draw is all-NaN — the GSS n=50k MPS refusal that
    survived the penalty port. Requiring per-chain descent keeps every iterate
    in the bounded sublevel set, so the fit converges to the same optimum the
    fp64 reference reaches.

    Returns ``(step (m,), accepted (m,), f1 (m,))`` — ``f1`` is each accepted
    chain's objective at its accepted step, reused by the caller as the next
    iteration's ``f0`` (one fewer NLL eval per iteration). No autograd.

    The slack is DTYPE-AWARE: descent is judged against the objective's own
    evaluation noise. In fp32 at survey n the NLL sum's rounding error is
    ~100-200x the fp64-appropriate ``1e-8`` slack (measured ~3e-5 absolute at
    n=37k vs slack 1.6e-7), so a near-converged chain would exhaust the budget
    on pure noise; ``100*eps`` of the compute dtype (fp64: keeps 1e-8) sits
    comfortably above it while still rejecting the ~1e3-magnitude oscillation
    increases this search exists to stop."""
    import torch

    with torch.no_grad():
        eps_slack = max(1e-8, 100.0 * torch.finfo(f0.dtype).eps)
        slack = eps_slack * (1.0 + f0.abs())
        step = torch.ones_like(f0)
        accepted = frozen.clone()
        for _ in range(_MAX_HALVINGS):
            trial = beta + step.unsqueeze(-1) * delta
            f1 = _pen_nll(trial, y, Xa, ridge_diag, beta0)
            ok = accepted | (torch.isfinite(f1) & (f1 <= f0 + slack))
            if bool(ok.all()):
                accepted = ok
                break
            step = torch.where(ok, step, 0.5 * step)
            accepted = ok
    return step, accepted, f1


def batched_logistic_irls(y, Xa):
    """Batched penalised IRLS, globalised by a per-chain halving line search
    (see module docstring). ``y`` (m, n_obs) 0/1, ``Xa`` (m, n_obs, k) WITH
    intercept. Returns ``(beta_hat (m, k), L (m, k, k))`` — the coefficient
    estimate and the lower Cholesky of the penalised Gram at ``beta_hat`` (for
    the posterior covariance ``G^{-1}``)."""
    import torch

    m, n, k = Xa.shape
    device = Xa.device

    ridge_diag, beta0 = _penalty_terms(y, Xa)
    beta = beta0.clone()

    # FP32 cannot reach the fp64 step tolerance (solve noise stalls the step
    # between ~3e-8 and ~4e-5 depending on the data, always > 1e-8, burning the
    # full budget); floor it at 1e-5, far below any statistically material
    # coefficient change (the polyreg/multinom floor). See module docstring.
    tol = _IRLS_TOL if Xa.dtype == torch.float64 else max(_IRLS_TOL, _IRLS_TOL_FP32)

    f0 = _pen_nll(beta, y, Xa, ridge_diag, beta0)                  # (m,)
    converged = torch.zeros(m, dtype=torch.bool, device=device)
    for _ in range(_MAX_IRLS_ITER):
        eta = (Xa @ beta.unsqueeze(-1)).squeeze(-1).clamp(-_ETA_CLIP, _ETA_CLIP)
        p = torch.sigmoid(eta)
        w = (p * (1.0 - p)).clamp_min(_W_FLOOR)
        # Penalised score: the ridge enters the gradient, not just the Gram —
        # at convergence beta solves the PENALISED equations (the 6.0.1 CPU
        # fix), so a separated fit shrinks toward beta0 instead of diverging.
        grad = (
            Xa.transpose(1, 2) @ (y - p).unsqueeze(-1)
            - (ridge_diag * (beta - beta0)).unsqueeze(-1)
        )                                                          # (m, k, 1)
        L = _pd_gram_cholesky(_ridged_gram(Xa, w, ridge_diag))
        delta = solve_spd(L, grad).squeeze(-1)                     # (m, k)
        # An fp32 device solve can transiently return a non-finite step (as in
        # the polr Newton). Zero it so it can never poison ``beta``; the line
        # search then re-evaluates from the last finite iterate.
        delta = torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))

        active = ~converged
        step, accepted, f1 = _halving_step(
            beta, delta, f0, y, Xa, ridge_diag, beta0, converged
        )
        # A chain with no observed-descent step within the budget FREEZES at
        # its last finite iterate (the polr search's semantics): near a
        # separated optimum the residual "increase" is objective evaluation
        # noise, not divergence, and the iterate is a valid fit. Poisoning
        # here (the earlier form) refused whole real-survey runs at n=50k
        # from noise alone. Genuine runaways never freeze finite: their trial
        # objectives increase by ~1e3, far past any slack, so their steps
        # shrink toward zero and the iterate stays in the bounded sublevel
        # set. Non-finite iterates cannot arise (steps are descent-accepted
        # from a finite start), so nothing silent slips through.
        update = (
            (active & accepted).to(Xa.dtype) * step
        ).unsqueeze(-1) * delta                                    # frozen/no-descent -> 0
        small = update.abs().amax(dim=1) < tol
        # Freeze converged chains so they stop drifting; matches the CPU break.
        beta = beta + update
        converged = converged | (active & (small | ~accepted))
        # f1 is the accepted-step objective for chains that moved; frozen
        # chains keep their (still valid) f0.
        f0 = torch.where(active & accepted, f1, f0)
        if bool(converged.all()):
            break

    # Posterior covariance at beta_hat: one final Gram/Cholesky at the estimate.
    eta = (Xa @ beta.unsqueeze(-1)).squeeze(-1).clamp(-_ETA_CLIP, _ETA_CLIP)
    p = torch.sigmoid(eta)
    w = (p * (1.0 - p)).clamp_min(_W_FLOOR)
    L = _pd_gram_cholesky(_ridged_gram(Xa, w, ridge_diag))
    return beta, L


def gpu_logreg_impute(y_obs, X_obs, X_mis, gen, *, donors=None, n_classes=None):
    """Batched Bayesian logistic-regression imputation (R ``logreg``).

    Parameters mirror the other GPU methods but batched: ``y_obs`` (m, n_obs) of
    0/1 class indices, ``X_obs`` (m, n_obs, q), ``X_mis`` (m, n_mis, q). Returns
    (m, n_mis) of 0/1 indices. ``donors`` and ``n_classes`` are accepted and
    ignored (binary is always 2-class) so the backend calls every categorical
    method with one uniform signature.

    The fit and draw run in FP64 where the device supports it
    (``discrete_glm_compute_dtype``): under (quasi-)separation the FP32 IRLS Gram
    goes near-singular, the Cholesky/solve returns a non-finite estimate, and the
    Bernoulli draw silently collapses every cell to 0 (``u < NaN`` is False) —
    which then corrupts every column using this one as a predictor. A non-finite
    predicted probability now yields NaN (never a silent 0), so a genuinely
    degenerate fit surfaces via the backend's end-of-sweep guard (Rule 1).
    """
    import torch

    out_dtype = X_obs.dtype
    compute_dtype = discrete_glm_compute_dtype(X_obs.device, out_dtype)
    X_obs = X_obs.to(compute_dtype)
    X_mis = X_mis.to(compute_dtype)
    y_obs = y_obs.to(compute_dtype)

    Xa = add_intercept(X_obs)
    beta_hat, L = batched_logistic_irls(y_obs, Xa)
    m, k = beta_hat.shape

    z = torch.randn((m, k), generator=gen, dtype=Xa.dtype, device=Xa.device)
    beta_star = beta_hat + apply_inv_factor_T(L, z.unsqueeze(-1)).squeeze(-1)

    Xa_mis = add_intercept(X_mis)
    eta = (Xa_mis @ beta_star.unsqueeze(-1)).squeeze(-1).clamp(-_ETA_CLIP, _ETA_CLIP)
    p = torch.sigmoid(eta)
    u = torch.rand(p.shape, generator=gen, dtype=p.dtype, device=p.device)
    out = (u < p).to(out_dtype)
    # Fail loud (Rule 1): non-finite p -> NaN, not a silent 0.
    return torch.where(torch.isfinite(p), out, torch.full_like(out, float("nan")))
