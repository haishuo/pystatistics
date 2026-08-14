"""
Separation/precision regression for the GPU discrete-GLM imputers (issue #8).

The GPU ``logreg`` (binary), ``polyreg`` (multinomial) and ``polr`` (ordinal)
fits are ill-conditioned under the (quasi-)separation chained equations routinely
induce. Run in FP32 the fit and posterior draw lose precision and the imputation
**silently collapses** every missing cell onto a single category — binary to
all-0 (``u < NaN`` is False), ordinal/nominal to category 0 (``argmax`` of an
all-False mask). On a mixed sweep the collapsed column then feeds a constant
predictor into every other column, so the damage is not local: it was measured
as a total-variation ~0.5 (vs ~0.1 in FP64) on the GSS mixed problem even after
the per-column ordinal fit was made finite.

The fix computes these discrete-GLM fits in FP64 where the device supports it
(``discrete_glm_compute_dtype``; MPS keeps FP32) and makes the samplers fail
loud — a non-finite probability yields NaN, never a silent category, so a
genuinely degenerate fit reaches the backend's end-of-sweep guard.

This module pins both halves: the fail-loud samplers (device-agnostic) and an
integrated on-device mixed sweep that must not collapse any column. The prior
suite exercised only balanced data and per-column fits, so it missed the
sweep-level collapse.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pystatistics.mice import mice
from pystatistics.mice.backends._gpu_logreg import gpu_logreg_impute
from pystatistics.mice.backends._gpu_polyreg import (
    _sample_categories as poly_sample_categories,
)
from pystatistics.mice.design import MICEDesign


def _accelerators() -> list[str]:
    devs = []
    if torch.cuda.is_available():
        devs.append("cuda")
    if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


# ----------------------------------------------------------------- fail-loud

class TestSamplersFailLoud:
    """The categorical samplers emit NaN (never a silent category 0/all-0) on
    non-finite probabilities, so a degenerate fit reaches the backend guard."""

    def test_polyreg_sampler_non_finite_row_yields_nan(self):
        probs = torch.ones(1, 3, 4) / 4.0
        probs[0, 1, :] = float("nan")
        gen = torch.Generator()
        gen.manual_seed(0)
        out = poly_sample_categories(probs, gen)
        assert torch.isnan(out[0, 1])
        assert torch.isfinite(out[0, 0]) and torch.isfinite(out[0, 2])

    def test_logreg_non_finite_predictor_yields_nan(self):
        """A non-finite predicted probability must surface as NaN, not a silent 0.
        (A non-finite fit drives ``eta`` -> NaN; ``eta`` is clamped, which tames
        +/-inf but propagates NaN, so a NaN predictor is the faithful proxy.)"""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 2)).astype(np.float64)
        y = (X[:, 0] > 0).astype(np.float64)
        X_mis = X[:5].copy()
        X_mis[0, 0] = np.nan  # -> NaN eta -> NaN p for row 0
        gen = torch.Generator()
        gen.manual_seed(0)
        out = gpu_logreg_impute(
            torch.tensor(y).unsqueeze(0),
            torch.tensor(X).unsqueeze(0),
            torch.tensor(X_mis).unsqueeze(0),
            gen,
        )
        assert torch.isnan(out[0, 0]), "non-finite predictor must yield NaN"
        assert torch.isfinite(out[0, 1:]).all()


class TestLogregIrlsGlobalisation:
    """The penalised IRLS must CONVERGE (bounded, stationary), not oscillate —
    the GSS n=50k regression. Real-survey designs carry structurally-zero
    dummy columns (levels that never co-occur with the target's observed rows)
    that dilute the mean-second-moment ridge, plus near-duplicate recode
    dummies that quasi-separate the target; the undamped penalised IRLS then
    oscillates (measured step growth 13 -> 2e5) and stops at the iteration cap
    on a wildly non-stationary iterate (|beta| ~ 1e5, penalised |grad| ~ 3e4,
    true optimum |beta| ~ 9). In fp32 a transient overshoot reaches inf and
    the imputation goes all-NaN — the GSS n=50k MPS refusal. The per-chain
    halving line search keeps every step descending; this fixture pins it in
    deterministic fp64 (no accelerator needed)."""

    @staticmethod
    def _diluted_separated_design():
        rng = np.random.default_rng(7)
        n = 30000
        a = (rng.random(n) < 0.08).astype(float)           # 8% recode dummy
        b = a.copy()
        flip_b = rng.random(n) < 0.002
        b[flip_b] = 1.0 - b[flip_b]                        # near-duplicate of a
        c = a.copy()
        flip_c = rng.random(n) < 0.003
        c[flip_c] = 1.0 - c[flip_c]                        # second near-duplicate
        zeros = np.zeros((n, 12))                           # structural zeros
        X = np.column_stack([a, b, c, zeros, rng.standard_normal((n, 8))])
        y = a.copy()
        noise = rng.random(n) < 0.001
        y[noise] = 1.0 - y[noise]                           # quasi-separation
        return y, X

    def test_gpu_irls_converges_bounded_and_stationary(self):
        from pystatistics.mice.backends._gpu_logreg import (
            batched_logistic_irls, _penalty_terms, _ETA_CLIP,
        )
        from pystatistics.mice.backends._gpu_linreg import add_intercept

        y, X = self._diluted_separated_design()
        m = 4
        Xa = add_intercept(torch.tensor(X, dtype=torch.float64).unsqueeze(0).repeat(m, 1, 1))
        yt = torch.tensor(y, dtype=torch.float64).unsqueeze(0).repeat(m, 1)
        beta, L = batched_logistic_irls(yt, Xa)
        assert torch.isfinite(beta).all() and torch.isfinite(L).all()
        assert float(beta.abs().max()) < 100, (
            f"non-stationary runaway: max|beta|={float(beta.abs().max()):.1f}"
        )
        ridge, b0 = _penalty_terms(yt, Xa)
        eta = (Xa @ beta.unsqueeze(-1)).squeeze(-1).clamp(-_ETA_CLIP, _ETA_CLIP)
        p = torch.sigmoid(eta)
        g = (Xa.transpose(1, 2) @ (yt - p).unsqueeze(-1)).squeeze(-1) - ridge * (beta - b0)
        assert float(g.abs().max()) < 1.0, (
            f"not a stationary point: penalised |grad|={float(g.abs().max()):.2f}"
        )

    def test_cpu_fit_converges_bounded(self):
        from pystatistics.mice.methods.logreg import _fit_logistic

        y, X = self._diluted_separated_design()
        beta, cov = _fit_logistic(y, X)
        assert np.isfinite(beta).all() and np.isfinite(cov).all()
        assert float(np.abs(beta).max()) < 100, (
            f"non-stationary runaway: max|beta|={float(np.abs(beta).max()):.1f}"
        )


# --------------------------------------------------------------------- on-device

def _mixed_separated_problem(seed: int, n: int = 3000):
    """A mixed-type problem engineered to drive the discrete fits into
    (quasi-)separation: a binary column near-perfectly ordered by a continuous
    predictor (logreg separation), and a 7-level ordered column with a near-empty
    intermediate category (polr). Numeric columns are well-conditioned. Returns
    ``(X, column_kinds)`` with ~20% missing completely at random per column."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    # binary near-separated by x
    b = (x > 0.8).astype(float)
    # ordered (7 levels) with a near-empty interior category, ordered by x
    cuts = np.quantile(x, [0.05, 0.07, 0.93, 0.95, 0.952, 0.99])
    o = np.digitize(x, cuts).astype(float)
    full = np.column_stack([b, o, x, z])
    kinds = ["binary", "ordered", "numeric", "numeric"]
    X = full.copy()
    # Missing only in the two discrete columns (the fits under test); keep the
    # numeric predictors fully observed so no row is all-missing and every
    # discrete fit has informative predictors.
    for j in (0, 1):
        X[rng.random(X.shape[0]) < 0.20, j] = np.nan
        X[:2, j] = full[:2, j]  # keep >=2 observed per column
    return X.astype(np.float64), kinds


@pytest.mark.skipif(not _accelerators(), reason="No CUDA/MPS device available")
class TestMixedSweepNoCollapse:
    """The default GPU sweep must not collapse any imputed column on a mixed
    problem with separated discrete columns — the issue #8 failure mode."""

    @pytest.mark.parametrize("device", _accelerators())
    def test_no_column_collapses(self, device):
        X, kinds = _mixed_separated_problem(seed=0)
        # MPS has no FP64; CUDA uses the double-precision path.
        gpu_backend = "gpu" if device == "mps" else "gpu_fp64"
        design = MICEDesign.from_array(X, method="auto", column_kinds=kinds)
        sol = mice(design, n_imputations=3, max_iter=5, seed=0, backend=gpu_backend)
        for col in sol.incomplete_columns:
            imp = sol.imputations(col)
            assert np.isfinite(imp).all(), f"col {col} produced non-finite imputations"
            # A collapsed fit imputes a single constant; a real fit varies.
            if kinds[col] in ("binary", "ordered"):
                assert np.unique(np.rint(imp)).size >= 2, (
                    f"col {col} ({kinds[col]}) collapsed to a single category"
                )

    @pytest.mark.parametrize("device", _accelerators())
    def test_complete_separation_by_recode_completes(self, device):
        """COMPLETE separation (not merely quasi): the binary target is a
        deterministic recode of an ORDERED predictor, missing in the SAME
        rows — the GSS survey structure ("case was retrieved" vs the ordered
        occupation-coding-status columns) that refused on MPS from 6.0.1 to
        6.1.1 and silently distorted on CUDA (fp64 kept the unpenalised fit
        finite but far from stationary, so the imputed marginal drifted to
        ~45% minority vs 8% observed on real GSS). The quasi-separated fixture
        above is too mild to trip either failure; this one requires the
        genuine (in-gradient) logreg penalty. The recode column is ordered —
        polr, penalised and line-search-globalised — deliberately NOT an
        unordered polyreg column, so each method's separation calibration is
        regressed in isolation (the polyreg twin is
        ``test_complete_separation_unordered_target_completes`` below)."""
        rng = np.random.default_rng(5)
        n = 4000
        cat = rng.choice([0.0, 1.0, 2.0], size=n, p=[0.92, 0.06, 0.02])
        b = (cat == 0.0).astype(float)                 # b == 1{cat == 0}
        X = np.column_stack([b, cat, rng.standard_normal(n), rng.standard_normal(n)])
        block = rng.random(n) < 0.25                   # one shared missing block
        X[block, 0] = np.nan
        X[block, 1] = np.nan
        kinds = ["binary", "ordered", "numeric", "numeric"]
        design = MICEDesign.from_array(X, method="auto", column_kinds=kinds)
        sol = mice(design, n_imputations=10, max_iter=8, seed=1, backend="gpu")
        imp = sol.imputations(0)                       # (m, n_mis)
        assert np.isfinite(imp).all()
        # POOLED majority must not collapse toward the minority class (the
        # 6.0.0 silent-collapse failure mode; majority ~92% observed). Not a
        # per-chain bound: the R-style posterior draw under separation
        # legitimately disperses individual chains on the CPU reference too.
        assert imp.mean() > 0.6

    @pytest.mark.parametrize("device", _accelerators())
    def test_complete_separation_unordered_target_completes(self, device):
        """COMPLETE separation of an UNORDERED (polyreg) target: two identical
        3-level categorical columns (an identity recode) missing in the SAME
        rows, so every chained fit of either on the other's dummies is
        completely separated in every class direction. The pre-fix
        ``_gpu_polyreg`` (Hessian-only ridge, unpenalised score) saturated the
        fp32 fit at |beta| ~1e26 and the draw collapsed every chain onto one
        category — finite, so invisible to the end-of-sweep NaN guard. The
        ported slope ridge + damped step keep it on the CPU optimum."""
        rng = np.random.default_rng(5)
        n = 2000
        cat = rng.choice([0.0, 1.0, 2.0], size=n, p=[0.92, 0.06, 0.02])
        X = np.column_stack(
            [cat, cat.copy(), rng.standard_normal(n), rng.standard_normal(n)]
        )
        block = rng.random(n) < 0.25                   # one shared missing block
        X[block, 0] = np.nan
        X[block, 1] = np.nan
        kinds = ["categorical", "categorical", "numeric", "numeric"]
        design = MICEDesign.from_array(X, column_kinds=kinds)
        gpu_backend = "gpu" if device == "mps" else "gpu_fp64"
        sol = mice(design, n_imputations=10, max_iter=8, seed=1, backend=gpu_backend)
        imp = sol.imputations(0)                       # (m, n_mis)
        assert np.isfinite(imp).all()
        m = imp.shape[0]
        constant_chains = sum(np.unique(imp[c]).size < 2 for c in range(m))
        assert constant_chains < m // 2, (
            f"{constant_chains}/{m} chains imputed a constant category"
        )
        assert (imp == 0.0).mean() > 0.5               # majority stays majority

    @pytest.mark.parametrize("device", _accelerators())
    def test_default_precision_matches_fp64(self, device):
        """On FP64-capable devices the discrete fits run in FP64 internally, so
        the *default* (FP32 sweep) ordered-category proportions track the FP64
        sweep — i.e. the default no longer silently degrades."""
        if device == "mps":
            pytest.skip("MPS has no FP64 to compare against")
        X, kinds = _mixed_separated_problem(seed=0)
        ordered = [j for j, k in enumerate(kinds) if k == "ordered"]

        def run(fp64):
            d = MICEDesign.from_array(X, method="auto", column_kinds=kinds)
            return mice(d, n_imputations=4, max_iter=5, seed=0,
                        backend="gpu_fp64" if fp64 else "gpu")

        sol32, sol64 = run(False), run(True)
        for col in ordered:
            K = int(np.nanmax(X[:, col])) + 1
            p32 = np.bincount(
                np.rint(sol32.imputations(col).ravel()).astype(int), minlength=K
            ).astype(float)
            p64 = np.bincount(
                np.rint(sol64.imputations(col).ravel()).astype(int), minlength=K
            ).astype(float)
            p32 /= p32.sum()
            p64 /= p64.sum()
            tv = 0.5 * float(np.abs(p32 - p64).sum())
            assert tv < 0.1, f"{device} col {col}: default-vs-fp64 TV={tv:.3f}"


# ------------------------------------------------------------- fit calibration

class TestPolyregPenalisedFitParity:
    """The batched GPU multinomial Newton computes the SAME penalised fit as
    the CPU ``methods/polyreg._fit_multinomial`` — the calibration contract of
    the 6.1.x fix (slope ridge in the score, intercepts unpenalised, damped
    step). Runs on CPU torch tensors in fp64, so it needs no accelerator and
    pins the calibration bit-tight on every platform."""

    @staticmethod
    def _fit_both(y, X, m=3):
        from pystatistics.mice._encode import add_intercept as np_add_intercept
        from pystatistics.mice.methods.polyreg import _fit_multinomial
        from pystatistics.mice.backends._gpu_linreg import add_intercept
        from pystatistics.mice.backends._gpu_polyreg import (
            batched_multinomial_newton,
        )

        K = int(y.max()) + 1
        coef, cov = _fit_multinomial(y.astype(np.intp), np_add_intercept(X), K)
        Xt = torch.as_tensor(X, dtype=torch.float64).unsqueeze(0).repeat(m, 1, 1)
        yt = torch.as_tensor(y, dtype=torch.float64).unsqueeze(0).repeat(m, 1)
        Xa = add_intercept(Xt)
        y_onehot = torch.zeros((m, Xa.shape[1], K), dtype=Xa.dtype)
        y_onehot.scatter_(2, yt.to(torch.int64).unsqueeze(-1), 1.0)
        beta, L = batched_multinomial_newton(y_onehot, Xa, K)
        return coef, cov, beta, L

    def test_well_identified_parity(self):
        rng = np.random.default_rng(0)
        n = 2000
        X = rng.standard_normal((n, 3))
        logits = np.column_stack(
            [0.8 * X[:, 0] - 0.3 * X[:, 1], -0.5 * X[:, 0] + 0.6 * X[:, 2],
             np.zeros(n)]
        )
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        y = np.array([rng.choice(3, p=pi) for pi in p], dtype=float)
        coef, cov, beta, L = self._fit_both(y, X)
        for c in range(beta.shape[0]):
            np.testing.assert_allclose(beta[c].numpy(), coef, atol=1e-10)
        Lm = L[0].numpy()
        np.testing.assert_allclose(np.linalg.inv(Lm @ Lm.T), cov, atol=1e-7)

    def test_complete_separation_parity_and_bounded(self):
        """Under complete separation both sides must land on the same FINITE
        penalised optimum (the pre-fix GPU diverged to ~1e26 in fp32 / ~2e10
        undamped in fp64 while the CPU delegation stalled elsewhere)."""
        rng = np.random.default_rng(0)
        n = 2000
        cat = rng.choice([0, 1, 2], size=n, p=[0.92, 0.06, 0.02])
        X = np.column_stack(
            [(cat == 1).astype(float), (cat == 2).astype(float),
             rng.standard_normal(n), rng.standard_normal(n)]
        )
        y = cat.astype(float)
        coef, cov, beta, L = self._fit_both(y, X)
        assert np.abs(coef).max() < 1e3                 # bounded, not divergent
        for c in range(beta.shape[0]):
            np.testing.assert_allclose(beta[c].numpy(), coef, atol=1e-9)


# ------------------------------------------------- line-search slack (fp32)

class TestLineSearchSlackDtypeAware:
    """The line-search acceptance slack must be judged against the objective's
    OWN evaluation noise — dtype-aware, the logreg 6.1.2 fix ported to
    polyreg/polr in 6.1.3. In fp32 at survey n (~50k) the penalised-NLL sum's
    rounding error (measured on MPS: sd ~8e-6, max ~2.3e-5) is 17-47x the
    fp64-appropriate ``1e-8*(1+f0)`` slack, so near-converged chains rejected
    truly-descending full steps on pure noise (measured: 798/2766 full-step
    rejections across 30 survey-scale polyreg sweeps, 13/36 on an ordered
    polr fixture — zero after the fix) — and the polyreg search POISONS on
    no-descent, making the noise hazard refusal-grade there.

    Deterministic mechanism pins (no accelerator, CPU fp32): hand the search
    an ``f0`` that sits a noise-scale amount BELOW the trial objective — the
    ratcheted state a noise-favoured accepted ``f0`` reaches — with a ZERO
    step, so every trial evaluates to exactly ``f0 + gap`` forever. A gap
    between the fp64 slack (1e-8) and the fp32 slack (100*eps ~ 1.2e-5)
    rejected every halving pre-fix; the dtype-aware slack accepts the full
    step. A genuine increase (gap far above any slack) must still reject, so
    the widened slack cannot mask the oscillation/saturation runaways the
    searches exist to stop."""

    # Gap coefficients relative to (1 + f0): between the two slacks / far above.
    _NOISE_GAP = 1e-6
    _GENUINE_GAP = 1e-2

    @staticmethod
    def _polyreg_inputs(m=2, n=400):
        from pystatistics.mice.backends._gpu_linreg import add_intercept
        from pystatistics.mice.backends._gpu_polyreg import _penalty_diag, _pen_nll

        rng = np.random.default_rng(0)
        K = 3
        X = torch.tensor(
            rng.standard_normal((n, 2)), dtype=torch.float32
        ).unsqueeze(0).repeat(m, 1, 1)
        y = torch.tensor(
            rng.choice(K, size=n).astype(np.float64), dtype=torch.float32
        ).unsqueeze(0).repeat(m, 1)
        Xa = add_intercept(X)
        y_onehot = torch.zeros((m, n, K), dtype=torch.float32)
        y_onehot.scatter_(2, y.to(torch.int64).unsqueeze(-1), 1.0)
        pen = _penalty_diag(Xa, K - 1)
        beta = torch.zeros((m, K - 1, Xa.shape[2]), dtype=torch.float32)
        f_true = _pen_nll(beta, y_onehot, Xa, pen)
        return beta, y_onehot, Xa, pen, f_true

    def test_polyreg_noise_gap_accepts_full_step(self):
        from pystatistics.mice.backends._gpu_polyreg import _halving_step

        beta, y_onehot, Xa, pen, f_true = self._polyreg_inputs()
        f0 = f_true - self._NOISE_GAP * (1.0 + f_true.abs())
        frozen = torch.zeros(beta.shape[0], dtype=torch.bool)
        step, accepted, _ = _halving_step(
            beta, torch.zeros_like(beta), f0, y_onehot, Xa, pen, frozen
        )
        assert bool(accepted.all()), "noise-scale gap rejected: chain would be poisoned"
        assert bool((step == 1.0).all())

    def test_polyreg_genuine_increase_still_rejected(self):
        from pystatistics.mice.backends._gpu_polyreg import _halving_step

        beta, y_onehot, Xa, pen, f_true = self._polyreg_inputs()
        f0 = f_true - self._GENUINE_GAP * (1.0 + f_true.abs())
        frozen = torch.zeros(beta.shape[0], dtype=torch.bool)
        _, accepted, _ = _halving_step(
            beta, torch.zeros_like(beta), f0, y_onehot, Xa, pen, frozen
        )
        assert not bool(accepted.any()), "genuine increase slipped past the slack"

    @staticmethod
    def _polr_inputs(m=2, n=400):
        from pystatistics.mice.backends._gpu_polr import _nll_per_chain

        rng = np.random.default_rng(1)
        K, q = 3, 2
        X = torch.tensor(
            rng.standard_normal((n, q)), dtype=torch.float32
        ).unsqueeze(0).repeat(m, 1, 1)
        y = torch.tensor(
            rng.choice(K, size=n).astype(np.float64), dtype=torch.float32
        ).unsqueeze(0).repeat(m, 1)
        slope_ridge = torch.full((m,), 0.1, dtype=torch.float32)
        params = torch.zeros((m, (K - 1) + q), dtype=torch.float32)
        f_true = _nll_per_chain(params, y, X, K, slope_ridge)
        return params, y, X, K, slope_ridge, f_true

    def test_polr_noise_gap_accepts_full_step(self):
        from pystatistics.mice.backends._gpu_polr import _backtracking_step

        params, y, X, K, slope_ridge, f_true = self._polr_inputs()
        f0 = f_true - self._NOISE_GAP * (1.0 + f_true.abs())
        frozen = torch.zeros(params.shape[0], dtype=torch.bool)
        step = _backtracking_step(
            params, torch.zeros_like(params), torch.zeros_like(f0), f0,
            y, X, K, slope_ridge, frozen,
        )
        assert bool((step == 1.0).all()), (
            "noise-scale gap rejected: chain would burn the backtrack budget "
            "and freeze pre-convergence"
        )

    def test_polr_genuine_increase_still_rejected(self):
        from pystatistics.mice.backends._gpu_polr import _backtracking_step

        params, y, X, K, slope_ridge, f_true = self._polr_inputs()
        f0 = f_true - self._GENUINE_GAP * (1.0 + f_true.abs())
        frozen = torch.zeros(params.shape[0], dtype=torch.bool)
        step = _backtracking_step(
            params, torch.zeros_like(params), torch.zeros_like(f0), f0,
            y, X, K, slope_ridge, frozen,
        )
        assert bool((step < 1e-12).all()), "genuine increase slipped past the slack"

    @staticmethod
    def _logreg_inputs(m=2, n=400):
        from pystatistics.mice.backends._gpu_linreg import add_intercept
        from pystatistics.mice.backends._gpu_logreg import _pen_nll, _penalty_terms

        rng = np.random.default_rng(2)
        X = torch.tensor(
            rng.standard_normal((n, 2)), dtype=torch.float32
        ).unsqueeze(0).repeat(m, 1, 1)
        y = torch.tensor(
            (rng.random(n) < 0.4).astype(np.float64), dtype=torch.float32
        ).unsqueeze(0).repeat(m, 1)
        Xa = add_intercept(X)
        ridge_diag, beta0 = _penalty_terms(y, Xa)
        beta = beta0.clone()
        f_true = _pen_nll(beta, y, Xa, ridge_diag, beta0)
        return beta, y, Xa, ridge_diag, beta0, f_true

    def test_logreg_noise_gap_accepts_full_step(self):
        """Pins the originating 6.1.2 logreg fix with the same mechanism."""
        from pystatistics.mice.backends._gpu_logreg import _halving_step

        beta, y, Xa, ridge_diag, beta0, f_true = self._logreg_inputs()
        f0 = f_true - self._NOISE_GAP * (1.0 + f_true.abs())
        frozen = torch.zeros(beta.shape[0], dtype=torch.bool)
        step, accepted, _ = _halving_step(
            beta, torch.zeros_like(beta), f0, y, Xa, ridge_diag, beta0, frozen
        )
        assert bool(accepted.all())
        assert bool((step == 1.0).all())

    def test_logreg_genuine_increase_still_rejected(self):
        from pystatistics.mice.backends._gpu_logreg import _halving_step

        beta, y, Xa, ridge_diag, beta0, f_true = self._logreg_inputs()
        f0 = f_true - self._GENUINE_GAP * (1.0 + f_true.abs())
        frozen = torch.zeros(beta.shape[0], dtype=torch.bool)
        _, accepted, _ = _halving_step(
            beta, torch.zeros_like(beta), f0, y, Xa, ridge_diag, beta0, frozen
        )
        assert not bool(accepted.any())


# ------------------------------------------------ shared Gram factor (6.1.3)

class TestCholeskyRidgedEscalationAndPoison:
    """``_gpu_linreg._cholesky_ridged`` (shared by the numeric draw and the
    polyreg Hessian) escalates its jitter per chain and explicitly poisons a
    chain that never factors — the logreg ``_pd_gram_cholesky`` lesson, ported
    sync-free. Pre-6.1.3 it added one fixed ``1e-8·scale`` jitter and returned
    ``cholesky_ex``'s output UNCHECKED: at survey n a failed factor can be
    FINITE garbage (measured on MPS: 5 of 10 failed polyreg-Hessian factors),
    which then flowed silently into the Newton step and the posterior draw.

    Deterministic CPU-fp32 pins via synthetic spectra: a healthy chain must
    keep its first-try factor bit-identically; a chain indefinite at the fp32
    accumulation-error scale (the GSS logreg coin-flip regime, between the
    base jitter and the 1e-2·scale cap) must factor finite via escalation; a
    chain indefinite far beyond the cap (a genuinely wrong matrix — degenerate
    design or the strided-matmul corruption in docs/GPU_NOTES.md) must come
    back all-NaN, never finite garbage."""

    @staticmethod
    def _spectrum_matrix(evals):
        """Symmetric fp32 matrix with the given eigenvalues (fixed basis)."""
        rng = np.random.default_rng(3)
        p = len(evals)
        Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
        M = (Q * np.asarray(evals)) @ Q.T
        return torch.tensor(0.5 * (M + M.T), dtype=torch.float32)

    def _batch(self):
        healthy = self._spectrum_matrix([5.0, 3.0, 2.0, 1.0])
        # mean diag ~2.5 -> jitter ladder ~2.5e-8, 2.5e-6, 2.5e-4, 2.5e-2:
        # -1e-4 factors at the second escalation, -1.0 never factors.
        marginal = self._spectrum_matrix([5.0, 3.0, 2.0, -1e-4])
        wrong = self._spectrum_matrix([5.0, 3.0, 2.0, -1.0])
        return torch.stack([healthy, marginal, wrong])

    def test_healthy_chain_bit_identical_first_try(self):
        from pystatistics.mice.backends._gpu_linreg import _cholesky_ridged

        G = self._batch()
        L = _cholesky_ridged(G)
        G0 = G[0:1]
        Gs0 = 0.5 * (G0 + G0.transpose(1, 2))
        jitter = torch.diagonal(Gs0, dim1=1, dim2=2).mean(dim=1).clamp_min(1.0) * 1e-8
        eye = torch.eye(4, dtype=torch.float32)
        expected, info = torch.linalg.cholesky_ex(Gs0 + jitter[:, None, None] * eye)
        assert int(info[0]) == 0
        assert torch.equal(L[0], expected[0])

    def test_marginally_indefinite_chain_recovered_by_escalation(self):
        from pystatistics.mice.backends._gpu_linreg import _cholesky_ridged

        G = self._batch()
        L = _cholesky_ridged(G)
        assert torch.isfinite(L[1]).all(), "escalation failed to rescue the chain"
        # The factor reconstructs G plus at most the capped jitter (1e-2*scale).
        scale = float(torch.diagonal(G[1]).mean())
        delta = (L[1] @ L[1].T) - G[1]
        assert float(delta.abs().max()) < 2e-2 * scale

    def test_never_pd_chain_poisoned_all_nan(self):
        from pystatistics.mice.backends._gpu_linreg import _cholesky_ridged

        G = self._batch()
        L = _cholesky_ridged(G)
        assert torch.isnan(L[2]).all(), (
            "a matrix indefinite beyond the jitter cap must poison, "
            "never return (possibly finite) garbage"
        )

    def test_all_healthy_batch_unchanged(self):
        from pystatistics.mice.backends._gpu_linreg import _cholesky_ridged

        rng = np.random.default_rng(4)
        A = torch.tensor(rng.standard_normal((3, 50, 6)), dtype=torch.float32)
        G = A.transpose(1, 2) @ A + 0.1 * torch.eye(6)
        L = _cholesky_ridged(G)
        assert torch.isfinite(L).all()
        recon = L @ L.transpose(1, 2)
        assert float((recon - 0.5 * (G + G.transpose(1, 2))).abs().max()) < 1e-4


class TestPmmFailLoud:
    """``gpu_pmm_impute`` must emit NaN, never a silently mis-matched donor,
    when the predicted means are non-finite (6.1.3). Donor copies are finite
    observed values, so unlike every other method a NaN-blind PMM stays
    invisible to the backend's end-of-sweep non-finite guard — the imputation
    is scrambled silently (measured: a chain with a poisoned fit returned
    fully finite donor copies). The mask is chain-wide for a non-finite
    observed prediction (the whole sorted donor ordering is invalid) and
    row-level for a non-finite missing-row prediction."""

    @staticmethod
    def _inputs(m=2, n=200, n_mis=6):
        from pystatistics.mice.backends._gpu_methods import gpu_pmm_impute

        rng = np.random.default_rng(5)
        X = torch.tensor(rng.standard_normal((n, 2)), dtype=torch.float32)
        y = (2.0 + X @ torch.tensor([1.0, -0.5]) +
             0.1 * torch.tensor(rng.standard_normal(n), dtype=torch.float32))
        Xm = torch.tensor(rng.standard_normal((n_mis, 2)), dtype=torch.float32)
        return (gpu_pmm_impute,
                y.unsqueeze(0).repeat(m, 1),
                X.unsqueeze(0).repeat(m, 1, 1),
                Xm.unsqueeze(0).repeat(m, 1, 1))

    def test_poisoned_chain_yields_nan_not_donors(self):
        impute, y, Xo, Xm = self._inputs()
        Xo[1, 0, 0] = float("nan")            # poisons chain 1's fit end-to-end
        gen = torch.Generator()
        gen.manual_seed(0)
        out = impute(y, Xo, Xm, gen)
        assert torch.isnan(out[1]).all(), "poisoned chain silently copied donors"
        assert torch.isfinite(out[0]).all()
        # healthy chain still returns genuine observed donor values
        assert all(float(v) in set(map(float, y[0])) for v in out[0])

    def test_nan_missing_row_masked_row_level(self):
        impute, y, Xo, Xm = self._inputs()
        Xm[0, 2, 0] = float("nan")            # chain 0, missing row 2 only
        gen = torch.Generator()
        gen.manual_seed(0)
        out = impute(y, Xo, Xm, gen)
        assert torch.isnan(out[0, 2])
        keep = torch.ones(out.shape[1], dtype=torch.bool)
        keep[2] = False
        assert torch.isfinite(out[0, keep]).all()
        assert torch.isfinite(out[1]).all()
