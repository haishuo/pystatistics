"""
MPS integrity diagnostics for the MICE GPU backend.

One job: let a user (or the release checklist) empirically test whether THEIR
torch/macOS/hardware combination exhibits the MPS allocator-state matmul
misread (docs/GPU_NOTES.md, "MPS strided-matmul buffer corruption") — the
upstream torch bug that silently corrupts ~1 chain fit in 20 at survey n on
affected builds.

The version classifier (``core.compute.device.mps_misread_status``) encodes
what was verified on the audit machine; this canary is the ground truth for
any other combination, because the bug was only ever *masked* upstream by an
allocator behavior change, not root-cause fixed — a future torch or a
different workload profile could resurface it.
"""

from __future__ import annotations

# The canary is a FIXED instrument: the one configuration whose healthy and
# corrupted outcomes are both measured (2026-08-13, M2 Max; see GPU_NOTES).
# Healthy: every chain matches the CPU fit to ~1e-3. Corrupted: at least one
# chain lands on a wrong finite optimum, deviation 8-43 in coefficient norm
# (measured 24-49 vs truth 15.6). The threshold sits far from both.
_CANARY_N = 50_000
_CANARY_M = 20
_CANARY_K = 3
_CANARY_SEED = 2000
_CANARY_DEVIATION_LIMIT = 1.0


def _canary_fixture():
    """The survey-shaped separated polyreg fixture that reliably exposed the
    misread on affected torch builds: a 92/6/2 three-level target that is a
    near-deterministic recode of its own dummy predictors, 12 structural-zero
    columns, 8 numerics, m chains differing in ~2% of predictor rows."""
    import numpy as np

    rng = np.random.default_rng(_CANARY_SEED)
    n, m = _CANARY_N, _CANARY_M
    cat = rng.choice([0.0, 1.0, 2.0], size=n, p=[0.92, 0.06, 0.02])
    d1 = (cat == 1.0).astype(float)
    d2 = (cat == 2.0).astype(float)
    b1 = d1.copy()
    f1 = rng.random(n) < 0.002
    b1[f1] = 1.0 - b1[f1]
    b2 = d1.copy()
    f2 = rng.random(n) < 0.003
    b2[f2] = 1.0 - b2[f2]
    X = np.column_stack(
        [d1, d2, b1, b2, np.zeros((n, 12)), rng.standard_normal((n, 8))]
    )
    Xs = np.repeat(X[None], m, axis=0)
    for c in range(1, m):
        mask = rng.random(n) < 0.02
        Xs[c, mask, 2] = 1.0 - Xs[c, mask, 2]
        mask = rng.random(n) < 0.02
        Xs[c, mask, 3] = 1.0 - Xs[c, mask, 3]
    return cat, Xs


def _fit(cat, Xs, device):
    """One batched polyreg Newton fit of the fixture on ``device`` (fp32 —
    the precision the misread was measured at). Returns beta on the host."""
    import torch

    from pystatistics.mice.backends._gpu_linreg import add_intercept
    from pystatistics.mice.backends._gpu_polyreg import batched_multinomial_newton

    Xa = add_intercept(torch.tensor(Xs, dtype=torch.float32, device=device))
    y = torch.tensor(cat, dtype=torch.float32, device=device)
    y = y.unsqueeze(0).repeat(Xs.shape[0], 1)
    m, n, _ = Xa.shape
    y_onehot = torch.zeros((m, n, _CANARY_K), dtype=torch.float32, device=device)
    y_onehot.scatter_(2, y.to(torch.int64).unsqueeze(-1), 1.0)
    beta, _L = batched_multinomial_newton(y_onehot, Xa, _CANARY_K)
    if device == "mps":
        torch.mps.synchronize()
    return beta.cpu()


def mps_matmul_canary() -> dict:
    """Fit the canary fixture on MPS and on the CPU (both fp32) and compare.

    Returns a dict: ``status`` ('clean' | 'corrupted'), ``max_deviation``
    (worst per-chain max|beta_mps - beta_cpu|), ``per_chain_deviation``
    (list of m floats), ``torch_version``, and the fixture shape. 'corrupted'
    means at least one chain's MPS fit landed off the CPU optimum by more
    than ``_CANARY_DEVIATION_LIMIT`` — on an affected build the deviation is
    8-43, on a clean one ~1e-3, so the verdict is unambiguous. A 'clean'
    result is evidence, not proof (the misread is allocator-state-dependent);
    a 'corrupted' result is proof of the bug on this combination.

    Deterministic given a torch build and device (fixed seed; no generator
    consumed). Runtime ~3 s (one MPS fit + one CPU reference fit at n=50k).
    Raises ``RuntimeError`` if MPS is unavailable.
    """
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "mps_matmul_canary tests the Apple-GPU (MPS) backend and needs "
            "an MPS device; on this machine there is nothing to test."
        )
    cat, Xs = _canary_fixture()
    beta_mps = _fit(cat, Xs, "mps")
    beta_cpu = _fit(cat, Xs, "cpu")
    dev = (beta_mps - beta_cpu).abs().amax(dim=(1, 2))
    max_dev = float(dev.max())
    return {
        "status": "corrupted" if max_dev > _CANARY_DEVIATION_LIMIT else "clean",
        "max_deviation": max_dev,
        "per_chain_deviation": [float(d) for d in dev],
        "torch_version": str(torch.__version__),
        "n": _CANARY_N,
        "m": _CANARY_M,
    }
