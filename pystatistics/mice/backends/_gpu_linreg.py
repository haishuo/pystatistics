"""
Batched Bayesian linear-regression posterior draw on the GPU.

This is the GPU counterpart of ``methods/_linreg.py``: the same Bayesian draw,
but vectorized over the ``m`` imputation chains as the leading batch dimension.
At a given sweep step every chain is fitting the *same* target column on the
*same* observed rows, but with different predictor values (each chain imputed
the other columns differently), so we solve ``m`` independent linear systems at
once with batched kernels. This path is shared by the CUDA and MPS devices.

Posterior draw (identical model to the CPU path), via a single Cholesky of the
ridged Gram ``G = X'X + ridge`` — no matrix inverse, no ``eigh``:

    G = L L'                                          batched Cholesky
    beta_hat = G^{-1} X'y = L^{-T} L^{-1} X'y         forward + back substitution
    df       = max(n_obs - n_params, 1)               integer, shared by chains
    sigma*   = sqrt( RSS / chi2(df) )                 chi2 = sum of df squared N(0,1)
    beta*    = beta_hat + sigma* * (L^{-T} z)         A = L^{-T}, A A' = G^{-1}

Applying ``L^{-1}`` is device-split via the shared dispatch in
``core.compute.linalg``: on MPS (whose triangular-solve kernels are ~100-300x
slower than matmul) ``L`` is inverted with the matmul-only blocked inverse and
applied by matmul; on CUDA/CPU the two back-substitutions (which share ``L'``)
are stacked into one fast triangular solve. The
chi-square is built from standard normals (``df`` is a non-negative integer) so
the only randomness source is the seeded ``torch.Generator`` — no reliance on
``torch.distributions``, which ignores explicit generators.

All randomness flows through the passed generator (CLAUDE.md Rule 6). Results
match the CPU reference distributionally, at the GPU/FP32 tolerance tier — not
bit-for-bit (different RNG, single precision).
"""

from __future__ import annotations

from dataclasses import dataclass

from pystatistics.core.compute.linalg import (
    batched_tri_inv_series,
    use_blocked_inverse,
)

# Matches the CPU ridge (methods/_linreg.py) — a tiny relative penalty keeping
# the batched Gram matrices invertible under FP32.
_DEFAULT_RIDGE = 1e-5

# Speculative jitter-escalation budget for _cholesky_ridged: ×100 per try from
# 1e-8·scale reaches 1e-2·scale — one spare decade past the measured fp32
# accumulation-error need (~5e-3·scale, the logreg GSS calibration). Fixed
# unroll (no data-dependent early exit) keeps the hot path sync-free; each try
# is one tiny (m, p, p) cholesky_ex.
_JITTER_ESCALATIONS = 3

# MPS dispatch threshold for the matmul-series inverse vs solve_triangular —
# torch-version-dependent. torch's MPS solve_triangular is a single but
# ~250x-slower-than-CUDA kernel; the series inverse is ~log2(p) fast matmuls
# but more launches. Under torch <= 2.12 the per-op dispatch cost made the
# small-n sweep launch-bound, so below ~n_obs 3000 the single solve_triangular
# kernel edged out the multi-op series (empirically tuned, m=100, q+1~20).
# torch 2.13 collapsed the MPS dispatch floor (~0.5-1 ms/op -> ~4 us/op), and
# the re-ablation (2026-08-12, M2 Max, same shapes) showed the series wins at
# EVERY n_obs — 6.75x at n_obs=1700, 3.2x at 6800, 1.95x at 17000 — so on
# torch >= 2.13 the threshold is bypassed (always series on MPS). A speed
# heuristic, not a correctness boundary. CUDA/CPU always use solve_triangular
# (fast there) — see use_blocked_inverse, which is MPS-only.
_SERIES_INV_MIN_NOBS = 3000


@dataclass
class BatchedLinRegDraw:
    """Batched point estimate + posterior draw (leading dim = m chains)."""

    beta_hat: "object"    # (m, q+1) tensor
    beta_draw: "object"   # (m, q+1) tensor
    sigma_draw: "object"  # (m,) tensor


def add_intercept(X):
    """Prepend a ones column to a batched predictor tensor (m, n, q) -> (m,n,q+1)."""
    import torch

    m, n, _ = X.shape
    ones = torch.ones((m, n, 1), dtype=X.dtype, device=X.device)
    return torch.cat([ones, X], dim=2)


def discrete_glm_compute_dtype(device, out_dtype):
    """Working precision for a discrete-outcome GLM fit + posterior draw: FP64
    where the device supports it (CUDA / CPU), else the caller's precision (MPS
    has no float64).

    The logistic / multinomial / proportional-odds fits are ill-conditioned under
    the (quasi-)separation chained equations routinely induce — a near-empty
    category or a perfectly ordered predictor drives thresholds/coefficients to
    large values and the information matrix to near-singular. In FP32 the fit and
    draw lose all precision and the imputation silently collapses every missing
    cell onto a single category (binary -> all-0, ordinal -> one level), which on
    a mixed sweep then corrupts every column that uses it as a predictor. These
    fits are per-column and small, so computing them in FP64 is cheap and makes
    the GPU imputation track the CPU reference (the numeric ``norm``/``pmm``
    methods are well-conditioned and stay at the sweep's precision). MPS keeps
    FP32 — it has no FP64 — but its matmul-only solve path is less exposed, and
    the sampled indices are always returned in the caller's dtype regardless."""
    import torch

    return out_dtype if device.type == "mps" else torch.float64


def batched_bayes_linreg_draw(
    y_obs,
    X_obs,
    gen,
    ridge: float = _DEFAULT_RIDGE,
) -> BatchedLinRegDraw:
    """Draw once from the Gaussian linear-model posterior for every chain.

    Parameters
    ----------
    y_obs : (m, n_obs) tensor
        Observed responses (identical across chains, batched for uniform ops).
    X_obs : (m, n_obs, q) tensor
        Observed predictors WITHOUT an intercept column (added internally).
    gen : torch.Generator
        Sole randomness source, on the same device as the tensors.
    ridge : float
        Relative ridge penalty on the diagonal of X'X.
    """
    import torch

    Xa = add_intercept(X_obs)                       # (m, n_obs, q+1)
    m, n_obs, n_params = Xa.shape
    dtype, device = Xa.dtype, Xa.device

    Xt = Xa.transpose(1, 2)                          # (m, q+1, n_obs)
    XtX = Xt @ Xa                                    # (m, q+1, q+1)

    diag = torch.diagonal(XtX, dim1=1, dim2=2)       # (m, q+1)
    diag_mean = diag.mean(dim=1).clamp_min(torch.finfo(dtype).tiny)  # (m,)
    eye = torch.eye(n_params, dtype=dtype, device=device)
    G = XtX + ridge * diag_mean[:, None, None] * eye  # ridged Gram, SPD (ridge>0)

    # Factor the ridged Gram once: G = L L'. We need beta_hat = G^-1 X'y and the
    # posterior factor A = L^-T (A A' = G^-1) — never the dense inverse, never
    # solve/inv/eigh. The Cholesky is sync-free; degenerate (collinear) input is
    # caught by the backend's end-of-sweep non-finite guard, not a per-step check.
    L = _cholesky_ridged(G)                          # (m, q+1, q+1) lower
    Lt = L.transpose(1, 2)

    # Draw both normal vectors up front, preserving the chi-then-beta RNG order.
    df = max(n_obs - n_params, 1)
    z_chi = torch.randn((m, df), generator=gen, dtype=dtype, device=device)
    z_beta = torch.randn((m, n_params), generator=gen, dtype=dtype, device=device)

    # HOW we apply L^-1 is device-, size-, and torch-version-split: MPS's
    # triangular-solve kernel is ~250x slower than its matmul, so we invert L
    # with the matmul-series inverse (Neumann doubling + 1 Newton step) and
    # apply by matmul — always on torch >= 2.13 (measured faster at every
    # n_obs), and above _SERIES_INV_MIN_NOBS on older torch, where the
    # dispatch-bound small-n sweep favoured the single solve_triangular kernel.
    # On CUDA/CPU triangular solves are fast — both keep solve_triangular.
    from pystatistics.core.compute.device import mps_native_kernels

    Xty = Xt @ y_obs.unsqueeze(-1)                   # (m, q+1, 1)
    if use_blocked_inverse(L) and (
        mps_native_kernels() or n_obs >= _SERIES_INV_MIN_NOBS
    ):
        Linv = batched_tri_inv_series(L)             # L^-1, matmul-series (MPS)
        Linvt = Linv.transpose(1, 2)
        beta_hat = (Linvt @ (Linv @ Xty)).squeeze(-1)    # G^-1 X'y via matmul
        Az = (Linvt @ z_beta.unsqueeze(-1)).squeeze(-1)  # A z, A = L^-T
    else:
        # The two upper solves share L', so stack them into one wide-RHS solve:
        #   beta_hat = L^-T (L^-1 X'y),  A z = L^-T z_beta.
        fwd = torch.linalg.solve_triangular(L, Xty, upper=False)
        upper = torch.linalg.solve_triangular(
            Lt, torch.cat([fwd, z_beta.unsqueeze(-1)], dim=2), upper=True
        )                                            # (m, q+1, 2)
        beta_hat = upper[..., 0]                      # (m, q+1)
        Az = upper[..., 1]                           # (m, q+1)

    resid = y_obs - (Xa @ beta_hat.unsqueeze(-1)).squeeze(-1)  # (m, n_obs)
    rss = (resid * resid).sum(dim=1)                 # (m,)
    # chi2(df) = sum of df squared standard normals (df is an integer).
    chi = (z_chi * z_chi).sum(dim=1).clamp_min(torch.finfo(dtype).tiny)  # (m,)
    sigma_draw = torch.where(
        rss > 0, torch.sqrt(rss / chi), torch.zeros_like(rss)
    )                                                # (m,)

    beta_draw = beta_hat + sigma_draw[:, None] * Az
    return BatchedLinRegDraw(
        beta_hat=beta_hat, beta_draw=beta_draw, sigma_draw=sigma_draw
    )


def _cholesky_ridged(G):
    """Lower Cholesky of each ridged Gram in the batch — sync-free, with
    per-chain jitter escalation and an explicit fail-loud poison.

    ``G = X'X + ridge·mean·I`` (ridge > 0) is positive definite in exact
    arithmetic. We symmetrize and add a *tiny* unconditional jitter
    (``1e-8·scale``, three orders below the ridge, so statistically negligible)
    to absorb FP32 rounding, then factor with ``cholesky_ex``.

    At survey n the base jitter is not always enough — the logreg
    ``_pd_gram_cholesky`` lesson: the fp32 Gram/Hessian accumulation error can
    exceed a diluted ridge floor, so numerical PD-ness becomes a per-chain
    coin flip (logreg measured one chain of 20, every sweep, on GSS n≈44k
    with a perfectly healthy fit). Chains whose first factor fails are retried
    at jitter ×100 per try up to ``1e-2·scale`` (the measured need there was
    ~5e-3·scale; a ~1e-2 relative widening of the posterior draw is
    negligible). Unlike polr's ``_pd_cholesky`` this escalation is
    SPECULATIVE — a fixed unroll of ``_JITTER_ESCALATIONS`` extra tiny
    factorizations selected per chain by ``where`` — because this helper is on
    the hot numeric-draw path, where the no-sync property below is
    load-bearing; first-try chains keep their factor bit-identically.

    A chain that fails ALL tries gets an explicitly all-NaN factor. The old
    code returned ``cholesky_ex``'s output unchecked, and on a failed
    factorization (``info > 0``) that output can be FINITE garbage — measured
    on MPS at n=50k: 5 of 10 failed polyreg-Hessian factors were finite — which
    then flows silently into the Newton step and the posterior draw. Matrices
    that far from PD are not rounding casualties but genuinely wrong inputs
    (degenerate designs, or the strided-matmul corruption documented in
    docs/GPU_NOTES.md); the NaN factor routes them to the backend's
    end-of-sweep guard (Rule 1) instead.

    Crucially there is still **no per-step host sync**: an earlier version
    called ``.item()`` and ``torch.any(info)`` on every sweep step (≈2 GPU↔CPU
    round-trips × ``maxit·p`` steps), which dominated the small-n sweep. All
    selection here stays on-device; genuinely degenerate chains surface at the
    backend's single end-of-sweep non-finite guard — one sync per sweep
    instead of hundreds. (Avoiding ``eigh``/``solve`` also keeps this valid
    and fast on MPS.)
    """
    import torch

    Gs = 0.5 * (G + G.transpose(1, 2))
    eye = torch.eye(Gs.shape[1], dtype=Gs.dtype, device=Gs.device)
    diag = torch.diagonal(Gs, dim1=1, dim2=2)
    jitter = diag.mean(dim=1).clamp_min(1.0) * 1e-8          # (m,), tensor (no .item())
    L, info = torch.linalg.cholesky_ex(Gs + jitter[:, None, None] * eye)
    ok = (info == 0) & torch.isfinite(L).flatten(1).all(dim=1)
    for _ in range(_JITTER_ESCALATIONS):
        jitter = jitter * 100.0
        L_try, info = torch.linalg.cholesky_ex(Gs + jitter[:, None, None] * eye)
        ok_try = (info == 0) & torch.isfinite(L_try).flatten(1).all(dim=1)
        take = ok_try & ~ok                     # first success wins, bit-stable
        L = torch.where(take[:, None, None], L_try, L)
        ok = ok | ok_try
    return torch.where(ok[:, None, None], L, torch.full_like(L, float("nan")))
