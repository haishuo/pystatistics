"""
Regression tests: the torch objectives' diagonal jitter must be
scale-relative, not absolute.

The batched objective evaluates the likelihood at ``Sigma_k + jitter*I`` on
each observed block. With the historical ABSOLUTE jitter (eps = 1e-6 on the
FP32 path), the minimizer of the jittered objective is ``Sigma_MLE - eps*I``,
so every fitted variance was biased downward by 1e-6 in absolute terms —
about -10% on data with true variances near 1e-5, silently accepted by every
pre- and post-fit guard. The jitter is now ``eps * var_j`` per variable
(constant during optimization, so the closed-form gradient remains exact),
making the bias ~ -eps in RELATIVE terms at any data scale.
"""

import numpy as np
import pytest

# Import pystatistics (scipy) BEFORE torch: importing torch first aborts with
# a duplicate-OpenMP-runtime error on conda macOS.
from pystatistics.mvnmle import mlest
from pystatistics.mvnmle.design import MVNDesign

pytest.importorskip("torch")

from pystatistics.mvnmle.backends.gpu import DirectMLEBackend  # noqa: E402


def _small_scale_problem(scale: float, n: int = 4000, seed: int = 7):
    """Complete-data problem with variances ~ scale**2 (closed-form MLE)."""
    rng = np.random.default_rng(seed)
    p = 3
    A = rng.standard_normal((p, p))
    cov = (A @ A.T + p * np.eye(p)) * scale**2
    mu = rng.standard_normal(p) * scale
    X = rng.multivariate_normal(mu, cov, size=n)
    mle_cov = np.cov(X, rowvar=False, ddof=0)  # complete-data MLE (n divisor)
    return X, mle_cov


class TestFP32JitterScaleRelative:
    @pytest.mark.parametrize("scale", [1.0, 3e-3])
    def test_fitted_variances_match_mle_relative(self, scale):
        X, mle_cov = _small_scale_problem(scale)
        backend = DirectMLEBackend(device='cpu', use_fp64=False)
        result = backend.solve(MVNDesign.from_array(X))
        fitted = np.diag(result.params.sigmahat)
        target = np.diag(mle_cov)
        rel_err = np.abs(fitted - target) / target
        # FP32 optimum precision is ~1e-3 relative; the old absolute jitter
        # produced ~-11% at scale 3e-3 (variances ~1e-5).
        assert np.all(rel_err < 5e-3), (
            f"scale={scale}: relative variance errors {rel_err} — "
            f"jitter bias is not scale-relative."
        )


class TestFP64PathUnaffectedAtUnitScale:
    def test_default_cpu_path_still_converges(self):
        X, mle_cov = _small_scale_problem(1.0)
        res = mlest(X)
        assert res.converged
        rel_err = np.abs(np.diag(res.sigmahat) - np.diag(mle_cov)) / np.diag(mle_cov)
        assert np.all(rel_err < 1e-6)
