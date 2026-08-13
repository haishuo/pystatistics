"""Characterization of the CPU ``polyreg`` slope-ridge behaviour (6.1.x).

``polyreg`` was the odd one out of the categorical family: ``logreg`` carries a
genuine in-gradient ridge and ``polr`` a slope objective ridge, but ``polyreg``
delegated to the UNPENALISED ``multinomial.multinom``. Under the complete
separation chained equations routinely induce (a categorical column that is a
deterministic recode of another — the GSS survey structure), that fit
"converged" wherever L-BFGS-B stalled (max |coef| ~51 on the audit fixture,
measured 2026-08-12), with a near-singular information whose inverse blew up
the posterior draw: the CPU imputed marginal on a 92/6/2 column came out
[0.18, 0.82, 0.00]. The batched GPU port of the same unpenalised fit collapsed
the column outright on fp32/MPS (all cells one category — silent corruption
invisible to the end-of-sweep non-finite guard, the 6.0.0 failure mode by a
different route).

The 6.1.x calibration gives ``polyreg`` the family treatment: the ``polr``
slope ridge (``lambda = 1e-5 * mean_slope_column_second_moment * n_obs``,
slopes only, per-class intercepts unpenalised) in the score AND the Hessian,
identical on CPU and GPU. This module pins the properties that calibration was
validated on, so they are not silently re-litigated:

  * **In-sample marginal fidelity.** The intercepts are unpenalised, so their
    score equations are exactly zero at the optimum and the average predicted
    category probability equals the observed marginal — even under complete
    separation (measured 4.4e-16). A regression that penalised the intercepts,
    or otherwise distorted the fit, would break this identity.

  * **Finite, bounded fit under separation** with a positive-definite
    penalised information — the ridge's actual job (audit fixture: |coef| 14.8
    vs the unpenalised divergence).

  * **Posterior-draw stability.** The penalised information keeps the draw
    bounded: drawn slopes stay within a small factor of the estimate and the
    drawn dominant-class predicted marginal has small spread (measured std
    0.0094 well-identified, 0.059 separated — vs the unpenalised draw that
    relocated the entire marginal).

  * **Agreement with R where R is well-posed.** On well-identified data the
    penalty is ~1e-5 relative: the fit matched ``nnet::multinom`` (reltol
    1e-12) to max |diff| 5.4e-5 (2026-08-12, nnet 7.3.20, n=2000, K=3). Under
    separation R's answer is an artifact of its BFGS stopping rule (no finite
    MLE exists; R reports |coef| ~33 where the old delegation reported ~51) —
    differences there are structural and do not favour either engine, exactly
    as established for the polr ridge. The R ``mice`` distributional gate in
    ``test_r_validation_categorical.py`` covers the imputation end-to-end.

See also ``tests/mice/test_polr_ridge_characterization.py`` (the precedent) and
``tests/mice/test_gpu_glm_separation.py`` / ``test_gpu_mps.py`` for the GPU
half of the fix.
"""

from __future__ import annotations

import numpy as np

from pystatistics.mice._encode import add_intercept
from pystatistics.mice.methods._draw import mvn_draw
from pystatistics.mice.methods.polyreg import (
    PolyregMethod,
    _fit_multinomial,
    _log_probs,
)

# Large but finite ceiling for the separated coefficients (the legitimate
# penalised fit is ~15; an unregularised divergence is orders larger).
_FINITE_BOUND = 1.0e3


def _well_identified_nominal(seed: int, n: int = 2000, K: int = 3):
    """A proper multinomial DGP with modest slopes and every class well
    populated — the regime where pystatistics, R, and the truth agree."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    logits = np.column_stack(
        [0.8 * X[:, 0] - 0.3 * X[:, 1], -0.5 * X[:, 0] + 0.6 * X[:, 2], np.zeros(n)]
    )
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(K, p=pi) for pi in p], dtype=np.intp)
    return y, X, K


def _separated_nominal(seed: int, n: int = 2000):
    """COMPLETE separation: the K=3 unordered target is a deterministic recode
    of a categorical predictor's dummies (the GSS structure) — every class
    perfectly predicted, imbalanced 92/6/2. The unpenalised MLE diverges; the
    slope ridge keeps it finite. Two inert numeric predictors round it out."""
    rng = np.random.default_rng(seed)
    cat = rng.choice([0, 1, 2], size=n, p=[0.92, 0.06, 0.02])
    d1 = (cat == 1).astype(float)
    d2 = (cat == 2).astype(float)
    X = np.column_stack([d1, d2, rng.standard_normal(n), rng.standard_normal(n)])
    return cat.astype(np.intp), X, 3


def _predicted_marginal(coef, Xa, K):
    return np.exp(_log_probs(Xa @ coef.T)).mean(axis=0)


def _observed_marginal(y, K):
    return np.bincount(y, minlength=K) / y.size


class TestInSampleMarginalFidelity:
    """The fit reproduces the observed marginal in-sample, in BOTH regimes —
    the intercept-stationarity property the slope ridge must not break."""

    def test_well_identified(self):
        y, X, K = _well_identified_nominal(seed=0)
        Xa = add_intercept(X)
        coef, _ = _fit_multinomial(y, Xa, K)
        np.testing.assert_allclose(
            _predicted_marginal(coef, Xa, K), _observed_marginal(y, K), atol=1e-8
        )

    def test_separated_still_faithful(self):
        """Holds to ~1e-15 even under complete separation: the intercepts are
        unpenalised, so the marginal identity survives the bounded slopes."""
        y, X, K = _separated_nominal(seed=0)
        Xa = add_intercept(X)
        coef, _ = _fit_multinomial(y, Xa, K)
        np.testing.assert_allclose(
            _predicted_marginal(coef, Xa, K), _observed_marginal(y, K), atol=1e-8
        )


class TestSeparatedFitFiniteAndBounded:
    """Under complete separation the ridge keeps the fit finite, bounded, and
    the penalised information PD — its actual purpose."""

    def test_fit_finite_bounded_pd(self):
        y, X, K = _separated_nominal(seed=0)
        coef, cov = _fit_multinomial(y, add_intercept(X), K)   # must not raise
        assert np.all(np.isfinite(coef))
        assert np.abs(coef).max() < _FINITE_BOUND
        assert float(np.linalg.eigvalsh(cov).min()) > 0.0


class TestDrawStability:
    """The posterior draw is well-conditioned and bounded — the pre-6.1.x
    draw from the unpenalised near-singular information relocated the whole
    imputed marginal (0.92 observed -> 0.18 imputed on the audit fixture)."""

    @staticmethod
    def _draw_stats(y, X, K, n_draws: int = 500):
        Xa = add_intercept(X)
        coef, cov = _fit_multinomial(y, Xa, K)
        dom = int(np.argmax(_observed_marginal(y, K)))
        rng = np.random.default_rng(0)
        doms, smax = [], []
        for _ in range(n_draws):
            bs = mvn_draw(coef.ravel(), cov, rng).reshape(K - 1, -1)
            doms.append(_predicted_marginal(bs, Xa, K)[dom])
            smax.append(np.abs(bs[:, 1:]).max())
        bmax = float(np.abs(coef[:, 1:]).max())
        return np.array(doms), np.array(smax), bmax

    def test_well_identified(self):
        y, X, K = _well_identified_nominal(seed=0)
        doms, smax, bmax = self._draw_stats(y, X, K)
        assert doms.std() < 0.05
        assert smax.max() < 3.0 * bmax

    def test_separated(self):
        # Measured std 0.059: wider than the well-identified 0.009 (the
        # separated information is legitimately flatter) but bounded — the
        # property under test is that the draw no longer blows up.
        y, X, K = _separated_nominal(seed=0)
        doms, smax, bmax = self._draw_stats(y, X, K)
        assert doms.std() < 0.10
        assert smax.max() < 3.0 * bmax


class TestWellIdentifiedMatchesWellPosedAnswer:
    """On a well-identified nominal with MCAR missingness — where pystatistics,
    R, and the truth coincide — the imputed marginal tracks the observed
    marginal (the R-agreement regime; the direct R mice gate lives in
    test_r_validation_categorical.py)."""

    def test_mcar_imputed_marginal_tracks_observed(self):
        y, X, K = _well_identified_nominal(seed=0)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        n_mis = int(0.3 * len(y))
        mis, obs = idx[:n_mis], idx[n_mis:]
        imp = PolyregMethod().impute(
            y[obs].astype(float), X[obs], X[mis], np.random.default_rng(1)
        )
        p_imp = np.bincount(np.rint(imp).astype(int), minlength=K) / imp.size
        tv = 0.5 * float(np.abs(p_imp - _observed_marginal(y, K)).sum())
        assert tv < 0.05, f"TV(imputed, observed) = {tv:.4f}"


class TestSeparatedImputationSane:
    """End-to-end on the separated fixture: the imputed marginal stays in the
    neighbourhood of the observed 92/6/2 split instead of relocating — the
    user-visible symptom the calibration fixes."""

    def test_imputed_marginal_near_observed(self):
        y, X, K = _separated_nominal(seed=0)
        rng = np.random.default_rng(0)
        mis = rng.random(len(y)) < 0.25
        pooled = np.zeros(K)
        for chain_seed in range(8):
            imp = PolyregMethod().impute(
                y[~mis].astype(float), X[~mis], X[mis],
                np.random.default_rng(chain_seed),
            )
            pooled += np.bincount(np.rint(imp).astype(int), minlength=K)
        pooled /= pooled.sum()
        tv = 0.5 * float(np.abs(pooled - _observed_marginal(y[~mis], K)).sum())
        assert tv < 0.10, f"TV(imputed, observed) = {tv:.4f} ({pooled})"
