"""GPU backend tests for MICE on Apple Silicon (MPS).

The MPS path shares the batched sweep with the CUDA path, diverging only in the
PMM donor search's insertion-rank op (``_gpu_methods._insertion_rank``: a
combined-sort merge-rank on MPS, ``searchsorted`` on CUDA). It runs FP32 only —
MPS has no float64 — so ``backend='gpu_fp64'`` is rejected at the dispatch
boundary (requires CUDA).

As with the CUDA suite, the MPS path is validated against the trusted CPU
implementation (itself validated against R), distributionally at the MPS/FP32
tolerance, plus the structural invariants. The categorical methods (logreg,
polyreg, polr) are additionally checked directly against R ``mice`` proportions on
the mixed fixture (``TestMpsCategoricalMatchesR``). All tests skip when MPS is
absent; the direct-R test also skips when the R fixtures are absent.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.mice import datasets, mice, pool
from pystatistics.mice.design import MICEDesign
from pystatistics.core.exceptions import ValidationError

_REF_DIR = Path(__file__).parent / "references"
_REF_JSON = _REF_DIR / "mice_categorical_reference.json"
_REF_CSV = _REF_DIR / "mice_categorical_data.csv"


def _mps_available() -> bool:
    try:
        import torch

        return torch.backends.mps.is_available()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _mps_available(), reason="No MPS device available")


@pytest.fixture(scope="module")
def miss():
    complete = datasets.make_gaussian_complete(200, seed=1)
    return datasets.make_mcar(complete, 0.25, seed=2)


class TestMpsBasics:
    def test_runs_on_mps(self, miss):
        sol = mice(miss, n_imputations=4, max_iter=5, seed=0, backend="gpu")
        assert "gpu" in sol.backend_name
        assert sol.info["device"] == "mps"
        assert sol.info["precision"] == "fp32"

    def test_completed_no_nan(self, miss):
        sol = mice(miss, n_imputations=4, max_iter=5, seed=0, backend="gpu")
        for d in sol.completed_datasets():
            assert not np.isnan(d).any()
            assert d.shape == miss.shape

    def test_observed_preserved(self, miss):
        sol = mice(miss, n_imputations=3, max_iter=4, seed=0, backend="gpu")
        observed = ~np.isnan(miss)
        for d in sol.completed_datasets():
            np.testing.assert_allclose(d[observed], miss[observed], rtol=1e-5, atol=1e-5)


class TestMpsFp64Rejected:
    def test_gpu_fp64_rejected_on_mps(self, miss):
        # MPS has no float64 — backend='gpu_fp64' must fail loud (requires CUDA),
        # not silently downgrade.
        with pytest.raises(RuntimeError, match="requires CUDA"):
            mice(miss, n_imputations=2, max_iter=3, seed=0, backend="gpu_fp64")


class TestMpsReproducibility:
    def test_same_seed_reproducible(self, miss):
        a = mice(miss, n_imputations=4, max_iter=5, method="pmm", seed=11, backend="gpu")
        b = mice(miss, n_imputations=4, max_iter=5, method="pmm", seed=11, backend="gpu")
        for da, db in zip(a.completed_datasets(), b.completed_datasets()):
            np.testing.assert_allclose(da, db, rtol=1e-5, atol=1e-6)

    def test_different_seed_differs(self, miss):
        a = mice(miss, n_imputations=4, max_iter=5, method="norm", seed=1, backend="gpu")
        b = mice(miss, n_imputations=4, max_iter=5, method="norm", seed=2, backend="gpu")
        assert any(
            not np.allclose(da, db)
            for da, db in zip(a.completed_datasets(), b.completed_datasets())
        )


class TestMpsPmmDonorProperty:
    def test_fp32_imputes_near_observed(self, miss):
        # Every PMM imputed value is a copy of an observed value (the donor
        # search is exact), to FP32 rounding — the hard correctness invariant.
        sol = mice(miss, n_imputations=5, max_iter=5, method="pmm", seed=0, backend="gpu")
        for j in sol.incomplete_columns:
            observed = miss[~np.isnan(miss[:, j]), j]
            imp = sol.imputations(j).ravel()
            dist = np.abs(imp[:, None] - observed[None, :]).min(axis=1)
            assert dist.max() < 1e-4


@pytest.mark.parametrize("method", ["pmm", "norm"])
class TestMpsMatchesCpu:
    def test_imputed_distribution_matches_cpu(self, miss, method):
        m = 30
        cpu = mice(miss, n_imputations=m, max_iter=10, method=method, seed=100, backend="cpu")
        gpu = mice(miss, n_imputations=m, max_iter=10, method=method, seed=100, backend="gpu")
        for j in gpu.incomplete_columns:
            cc = cpu.imputations(j).ravel()
            cg = gpu.imputations(j).ravel()
            assert abs(cg.mean() - cc.mean()) < 0.15, f"col {j} mean drift"
            assert abs(cg.std() - cc.std()) < 0.15, f"col {j} sd drift"

    def test_pooled_regression_matches_cpu(self, miss, method):
        def pooled(sol):
            est, var = [], []
            for d in sol.completed_datasets():
                X = np.column_stack([np.ones(len(d)), d[:, 1], d[:, 2]])
                y = d[:, 0]
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                resid = y - X @ beta
                s2 = resid @ resid / (len(d) - 3)
                est.append(beta)
                var.append(np.diag(s2 * np.linalg.inv(X.T @ X)))
            return pool(np.array(est), np.array(var), df_complete=len(miss) - 3)

        m = 30
        cpu = pooled(mice(miss, n_imputations=m, max_iter=10, method=method, seed=7, backend="cpu"))
        gpu = pooled(mice(miss, n_imputations=m, max_iter=10, method=method, seed=7, backend="gpu"))
        np.testing.assert_allclose(
            np.asarray(gpu.estimate), np.asarray(cpu.estimate), atol=0.1
        )


class TestMpsDegenerate:
    def test_collinear_predictors_never_silent_nan(self):
        # The batched draw runs sync-free and defers fail-loud to the backend's
        # end-of-sweep non-finite guard. With heavily collinear predictors the
        # ridge keeps the Gram solvable; either way the result must be finite or
        # the run must fail loud — never a silently-returned NaN.
        from pystatistics.core.exceptions import ValidationError

        rng = np.random.default_rng(0)
        n = 300
        x = rng.standard_normal((n, 1))
        X = np.column_stack([
            x,                                    # fully observed anchor
            x + 1e-9 * rng.standard_normal((n, 1)),  # near-duplicate of col 0
            rng.standard_normal((n, 3)),
        ])
        mask = rng.random(X.shape) < 0.2
        mask[:, 0] = False
        X[mask] = np.nan
        try:
            sol = mice(X, n_imputations=4, max_iter=5, method="pmm", seed=0, backend="gpu")
        except ValidationError:
            return  # acceptable: failed loud rather than returning garbage
        for d in sol.completed_datasets():
            assert np.isfinite(d).all()


class TestMpsCategoricalPredictors:
    """Numeric targets imputed from categorical predictors (dummy-encoded on
    GPU). Categorical *targets* are not yet supported and must be refused."""

    @staticmethod
    def _mixed_design():
        rng = np.random.default_rng(0)
        n = 600
        cat = rng.integers(0, 3, n).astype(float)          # categorical predictor
        mu = np.array([0.0, 2.5, -1.5])[cat.astype(int)]    # targets depend on it
        X = np.column_stack([
            cat,
            mu + rng.normal(0, 1, n),
            0.5 * mu + rng.normal(0, 1, n),
        ])
        mask = rng.random(X.shape) < 0.2
        mask[:, 0] = False                                  # categorical stays complete
        X[mask] = np.nan
        return MICEDesign.from_array(
            X, method="pmm", column_kinds=["categorical", "numeric", "numeric"]
        )

    def test_categorical_predictor_matches_cpu(self):
        d = self._mixed_design()
        assert d.has_categorical
        cpu = mice(d, n_imputations=30, max_iter=10, seed=1, backend="cpu")
        gpu = mice(d, n_imputations=30, max_iter=10, seed=1, backend="gpu")
        assert gpu.info["device"] == "mps"
        # Matching CPU (which dummy-encodes the categorical) confirms the GPU
        # uses the categorical predictor correctly rather than ignoring it.
        for j in gpu.incomplete_columns:
            c = cpu.imputations(j).ravel()
            g = gpu.imputations(j).ravel()
            assert abs(g.mean() - c.mean()) < 0.15, f"col {j} mean drift"
            assert abs(g.std() - c.std()) < 0.15, f"col {j} sd drift"
        for dset in gpu.completed_datasets():
            assert not np.isnan(dset).any()

    def test_method_without_gpu_kernel_refused(self, monkeypatch):
        # Every registered method now has a GPU kernel, so to exercise the
        # backend's fail-loud guard we drop one kernel and confirm a column
        # assigned that method is refused (not silently downgraded).
        from pystatistics.mice.backends import _gpu_methods

        monkeypatch.delitem(_gpu_methods.GPU_METHODS, "pmm")
        rng = np.random.default_rng(1)
        n = 200
        X = np.column_stack([rng.normal(0, 1, n), rng.normal(0, 1, n)])
        X[rng.random(n) < 0.2, 0] = np.nan
        d = MICEDesign.from_array(X, method="pmm")
        with pytest.raises(ValidationError, match="no GPU implementation"):
            mice(d, n_imputations=4, max_iter=3, seed=0, backend="gpu")


class TestMpsLogregBinaryTarget:
    """Binary (incomplete) targets imputed on GPU via batched IRLS logistic
    regression. Validated against the trusted CPU ``logreg`` distributionally at
    the MPS/FP32 tolerance, plus the structural invariants (valid codes, no NaN,
    determinism). Multinomial/ordinal targets remain refused (a later stage)."""

    @staticmethod
    def _binary_design(codes=(0.0, 1.0), seed=0):
        rng = np.random.default_rng(seed)
        n = 600
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        eta = 0.4 + 1.1 * x1 - 0.7 * x2
        p = 1.0 / (1.0 + np.exp(-eta))
        lo, hi = codes
        b = np.where(rng.random(n) < p, hi, lo)
        X = np.column_stack([b, x1, x2])
        # Binary target and one numeric column incomplete; keep x1 fully observed
        # so every row retains an observed predictor (no all-NaN rows).
        X[rng.random(n) < 0.25, 0] = np.nan
        X[rng.random(n) < 0.20, 2] = np.nan
        return MICEDesign.from_array(
            X, column_kinds=["binary", "numeric", "numeric"]
        )

    def test_runs_on_mps_with_logreg(self):
        d = self._binary_design()
        assert d.method_for(0) == "logreg"
        sol = mice(d, n_imputations=4, max_iter=5, seed=0, backend="gpu")
        assert sol.info["device"] == "mps"

    def test_imputed_values_are_valid_codes(self):
        # Arbitrary codes (not 0/1) exercise the batched code<->index mapping.
        d = self._binary_design(codes=(2.0, 5.0), seed=1)
        sol = mice(d, n_imputations=10, max_iter=8, seed=3, backend="gpu")
        allowed = set(d.levels_for(0).tolist())
        assert set(np.unique(sol.imputations(0)).tolist()).issubset(allowed)

    def test_completed_no_nan(self):
        d = self._binary_design()
        for dset in mice(d, n_imputations=5, max_iter=6, seed=2, backend="gpu").completed_datasets():
            assert not np.isnan(dset).any()

    def test_proportion_matches_cpu(self):
        d = self._binary_design()
        m = 40
        cpu = mice(d, n_imputations=m, max_iter=10, seed=7, backend="cpu")
        gpu = mice(d, n_imputations=m, max_iter=10, seed=7, backend="gpu")
        # Imputed-category proportion of the binary target: independent RNG
        # streams, so distributional (not bit-for-bit) agreement.
        c1 = cpu.imputations(0).ravel().mean()
        g1 = gpu.imputations(0).ravel().mean()
        assert abs(c1 - g1) < 0.06, f"binary proportion drift: cpu={c1:.3f} gpu={g1:.3f}"

    def test_numeric_codraw_matches_cpu(self):
        # The numeric column imputed alongside the binary target should still
        # track CPU — confirms the mixed sweep (logreg + pmm) stays consistent.
        d = self._binary_design()
        m = 40
        cpu = mice(d, n_imputations=m, max_iter=10, seed=7, backend="cpu")
        gpu = mice(d, n_imputations=m, max_iter=10, seed=7, backend="gpu")
        c = cpu.imputations(2).ravel()
        g = gpu.imputations(2).ravel()
        assert abs(c.mean() - g.mean()) < 0.15
        assert abs(c.std() - g.std()) < 0.15

    def test_same_seed_reproducible(self):
        d = self._binary_design()
        a = mice(d, n_imputations=6, max_iter=6, seed=11, backend="gpu")
        b = mice(d, n_imputations=6, max_iter=6, seed=11, backend="gpu")
        for da, db in zip(a.completed_datasets(), b.completed_datasets()):
            np.testing.assert_array_equal(da, db)


class TestMpsPolyregNominalTarget:
    """Unordered categorical (incomplete) targets imputed on GPU via batched
    multinomial-logit Newton. Validated against the trusted CPU ``polyreg``
    distributionally at the MPS/FP32 tolerance, plus structural invariants.
    Ordinal (polr) targets remain refused (a later stage)."""

    @staticmethod
    def _nominal_design(K=4, codes=None, seed=0):
        rng = np.random.default_rng(seed)
        n, q = 700, 3
        X = rng.standard_normal((n, q))
        B = rng.standard_normal((K - 1, q + 1)) * 1.1
        Xa = np.column_stack([np.ones(n), X])
        eta = np.column_stack([Xa @ B.T, np.zeros(n)])
        eta -= eta.max(1, keepdims=True)
        p = np.exp(eta)
        p /= p.sum(1, keepdims=True)
        y = np.array([rng.choice(K, p=p[i]) for i in range(n)], dtype=int)
        lev = np.arange(K, dtype=float) if codes is None else np.asarray(codes, float)
        ynom = lev[y]
        data = np.column_stack([ynom, X])
        data[rng.random(n) < 0.25, 0] = np.nan      # only the nominal target incomplete
        kinds = ["categorical"] + ["numeric"] * q
        return MICEDesign.from_array(data, column_kinds=kinds)

    def test_runs_on_mps_with_polyreg(self):
        d = self._nominal_design()
        assert d.method_for(0) == "polyreg"
        sol = mice(d, n_imputations=4, max_iter=5, seed=0, backend="gpu")
        assert sol.info["device"] == "mps"

    def test_imputed_values_are_valid_codes(self):
        # Arbitrary, non-consecutive codes exercise the batched code<->index map.
        d = self._nominal_design(K=4, codes=[10.0, 20.0, 30.0, 40.0], seed=1)
        sol = mice(d, n_imputations=10, max_iter=8, seed=3, backend="gpu")
        allowed = set(d.levels_for(0).tolist())
        assert set(np.unique(sol.imputations(0)).tolist()).issubset(allowed)

    def test_completed_no_nan(self):
        d = self._nominal_design()
        for dset in mice(d, n_imputations=5, max_iter=6, seed=2, backend="gpu").completed_datasets():
            assert not np.isnan(dset).any()

    def test_category_proportions_match_cpu(self):
        d = self._nominal_design()
        m = 40
        cpu = mice(d, n_imputations=m, max_iter=10, seed=7, backend="cpu")
        gpu = mice(d, n_imputations=m, max_iter=10, seed=7, backend="gpu")
        levels = d.levels_for(0)
        ci = cpu.imputations(0).ravel()
        gi = gpu.imputations(0).ravel()
        cp = np.array([np.mean(ci == lv) for lv in levels])
        gp = np.array([np.mean(gi == lv) for lv in levels])
        # Per-category proportion agreement (independent RNG streams).
        assert np.max(np.abs(cp - gp)) < 0.06, f"cpu={np.round(cp,3)} gpu={np.round(gp,3)}"

    def test_same_seed_reproducible(self):
        d = self._nominal_design()
        a = mice(d, n_imputations=6, max_iter=6, seed=11, backend="gpu")
        b = mice(d, n_imputations=6, max_iter=6, seed=11, backend="gpu")
        for da, db in zip(a.completed_datasets(), b.completed_datasets()):
            np.testing.assert_array_equal(da, db)

    def test_three_class_target(self):
        # K=3 (smallest multinomial) alongside the K=4 default — guards the
        # block-Hessian assembly across class counts.
        d = self._nominal_design(K=3, seed=4)
        sol = mice(d, n_imputations=8, max_iter=8, seed=1, backend="gpu")
        allowed = set(d.levels_for(0).tolist())
        assert set(np.unique(sol.imputations(0)).tolist()).issubset(allowed)
        for dset in sol.completed_datasets():
            assert not np.isnan(dset).any()


class TestMpsPolrOrderedTarget:
    """Ordered categorical (incomplete) targets imputed on GPU via batched
    proportional-odds Newton in the raw (unconstrained) threshold
    parameterization — thresholds stay ordered by construction. Validated against
    the trusted CPU ``polr`` distributionally at the MPS/FP32 tolerance, plus
    structural invariants."""

    @staticmethod
    def _ordered_design(K=4, codes=None, seed=0):
        rng = np.random.default_rng(seed)
        n, q = 700, 3
        X = rng.standard_normal((n, q))
        beta = rng.standard_normal(q) * 0.9
        eta = X @ beta
        cuts = np.linspace(-1.2, 1.2, K - 1)
        u = rng.uniform(size=n)
        cum = 1.0 / (1.0 + np.exp(-(cuts[None, :] - eta[:, None])))
        y = (u[:, None] > cum).sum(axis=1).astype(int)
        for lv in range(K):                          # guarantee all levels present
            if not np.any(y == lv):
                y[lv] = lv
        lev = np.arange(K, dtype=float) if codes is None else np.asarray(codes, float)
        data = np.column_stack([lev[y], X])
        data[rng.random(n) < 0.25, 0] = np.nan       # only the ordered target incomplete
        kinds = ["ordered"] + ["numeric"] * q
        return MICEDesign.from_array(data, column_kinds=kinds)

    def test_runs_on_mps_with_polr(self):
        d = self._ordered_design()
        assert d.method_for(0) == "polr"
        sol = mice(d, n_imputations=4, max_iter=4, seed=0, backend="gpu")
        assert sol.info["device"] == "mps"

    def test_imputed_values_are_valid_codes(self):
        # Non-0-based ordered codes exercise the batched code<->index mapping.
        d = self._ordered_design(K=4, codes=[3.0, 6.0, 9.0, 12.0], seed=1)
        sol = mice(d, n_imputations=8, max_iter=6, seed=3, backend="gpu")
        allowed = set(d.levels_for(0).tolist())
        assert set(np.unique(sol.imputations(0)).tolist()).issubset(allowed)

    def test_completed_no_nan(self):
        d = self._ordered_design()
        for dset in mice(d, n_imputations=4, max_iter=5, seed=2, backend="gpu").completed_datasets():
            assert not np.isnan(dset).any()

    def test_category_proportions_match_cpu(self):
        d = self._ordered_design()
        m = 30
        cpu = mice(d, n_imputations=m, max_iter=10, seed=7, backend="cpu")
        gpu = mice(d, n_imputations=m, max_iter=10, seed=7, backend="gpu")
        levels = d.levels_for(0)
        ci = cpu.imputations(0).ravel()
        gi = gpu.imputations(0).ravel()
        cp = np.array([np.mean(ci == lv) for lv in levels])
        gp = np.array([np.mean(gi == lv) for lv in levels])
        # Proportions are insensitive to the threshold-draw covariance; the
        # covariance itself is pinned in tests/mice/test_gpu_polr_draw.py.
        assert np.max(np.abs(cp - gp)) < 0.03, f"cpu={np.round(cp,3)} gpu={np.round(gp,3)}"

    def test_imputed_codes_preserve_order_monotonicity(self):
        # Ordered structure should be reflected: the imputed marginal should not
        # collapse to a single level (a sanity check that the model uses X).
        d = self._ordered_design()
        sol = mice(d, n_imputations=10, max_iter=8, seed=4, backend="gpu")
        assert len(np.unique(sol.imputations(0))) >= 3

    def test_same_seed_reproducible(self):
        d = self._ordered_design()
        a = mice(d, n_imputations=5, max_iter=5, seed=11, backend="gpu")
        b = mice(d, n_imputations=5, max_iter=5, seed=11, backend="gpu")
        for da, db in zip(a.completed_datasets(), b.completed_datasets()):
            np.testing.assert_array_equal(da, db)


@pytest.mark.skipif(
    not (_REF_JSON.exists() and _REF_CSV.exists()),
    reason="R categorical fixtures absent (run generate_categorical_fixtures.R)",
)
class TestMpsCategoricalMatchesR:
    """Direct GPU-vs-R validation on the mixed categorical fixture. With logreg,
    polyreg and polr all on GPU, the whole bin/nom/ord dataset runs on MPS, so we
    can check imputed category proportions against R ``mice`` 3.19.0 directly
    (not only transitively via CPU). Distributional agreement (independent RNG)."""

    _COL_BY_NAME = {"bin": 1, "nom": 2, "ord": 3}

    @pytest.fixture(scope="class")
    def r_and_gpu(self):
        ref = json.load(open(_REF_JSON))
        matrix = np.genfromtxt(_REF_CSV, delimiter=",", skip_header=1)
        kinds = ["numeric", "binary", "categorical", "ordered"]
        design = MICEDesign.from_array(matrix, column_kinds=kinds)
        meta = ref["meta"]
        sol = mice(
            design, n_imputations=meta["m"], max_iter=meta["maxit"], seed=20260614, backend="gpu"
        )
        return ref, design, sol

    @pytest.mark.parametrize("name", ["bin", "nom", "ord"])
    def test_proportions_match_r(self, r_and_gpu, name):
        ref, design, sol = r_and_gpu
        assert sol.info["device"] == "mps"
        col = self._COL_BY_NAME[name]
        levels = [int(lv) for lv in ref[name]["levels"]]
        imp = sol.imputations(col).ravel()
        counts = np.array([np.sum(imp == float(lv)) for lv in levels], dtype=float)
        ours = counts / counts.sum()
        r_prop = np.asarray(ref[name]["proportions"], dtype=float)
        np.testing.assert_allclose(ours, r_prop, atol=0.06, err_msg=(
            f"{name}: gpu={np.round(ours,3)} R={np.round(r_prop,3)}"
        ))


class TestMpsScales:
    def test_large_n_matches_cpu_distribution(self):
        complete = datasets.make_gaussian_complete(8000, seed=2)
        miss = datasets.make_mcar(complete, 0.2, seed=3)
        cpu = mice(miss, n_imputations=20, max_iter=8, method="pmm", seed=5, backend="cpu")
        gpu = mice(miss, n_imputations=20, max_iter=8, method="pmm", seed=5, backend="gpu")
        for j in gpu.incomplete_columns:
            assert abs(gpu.imputations(j).mean() - cpu.imputations(j).mean()) < 0.1


@pytest.mark.skipif(not _mps_available(), reason="MPS not available")
class TestGpuCategoricalNoSilentCollapse:
    """GPU categorical fits must fail loud, never silently collapse — 6.0.x.

    A degenerate fp32 fit emits an all-NaN sentinel; the old
    ``indices_to_codes`` cast NaN to level 0, silently collapsing the column.
    The sentinel now propagates so the backend's end-of-sweep guard raises.
    """

    def test_degenerate_categorical_raises_not_collapses(self):
        rng = np.random.default_rng(3)
        n = 120
        x0 = rng.standard_normal(n) * 1e20  # extreme-scale degenerate predictor
        x1 = rng.standard_normal(n)
        y = rng.integers(0, 3, n).astype(float)
        data = np.column_stack([x0, x1, y])
        data[rng.random(n) < 0.3, 2] = np.nan
        des = MICEDesign.from_array(
            data, method="auto",
            column_kinds=["numeric", "numeric", "categorical"],
        )
        with pytest.raises(ValidationError, match="non-finite"):
            mice(des, seed=3, n_imputations=6, max_iter=4, backend="gpu")

    def test_indices_to_codes_propagates_nan(self):
        import torch
        from pystatistics.mice.backends._gpu_encode import indices_to_codes
        levels = torch.tensor([0.0, 1.0, 2.0])
        idx = torch.tensor([0.0, float("nan"), 2.0])
        out = indices_to_codes(idx, levels)
        assert torch.isnan(out[1])           # sentinel preserved, not -> level 0
        assert out[0].item() == 0.0 and out[2].item() == 2.0


class TestMpsLogregCompleteSeparation:
    """A completely separated binary target must fit finite on MPS fp32 — 6.1.x.

    Real-survey regression (GSS): a binary "case was retrieved" flag is a
    deterministic recode of a categorical coding-status column, so every
    chained-equations fit of that column is COMPLETELY separated by the
    recode's dummies. The GPU IRLS carried only the pre-6.0.1 Hessian ridge
    (unpenalised score), so beta diverged, the fp32 fit went non-finite, and
    the NaN sentinel — correctly propagated since 6.0.1 — had the end-of-sweep
    guard refuse the whole run (~30% of chain-sweeps degenerate at m=20,
    maxit=10). With the CPU's genuine penalty ported, the fit stays finite and
    the run completes with the observed marginal reproduced, matching the CPU
    reference on the same data.
    """

    @staticmethod
    def _separated_design(seed=0):
        # Mirror the GSS structure: binary col 0 is a deterministic recode of
        # the 3-level column 1 (col1 > 0  <=>  col0 == 0), imbalanced ~8/92,
        # missing in the SAME rows (one missingness block), plus numeric noise.
        rng = np.random.default_rng(seed)
        n = 2000
        cat = rng.choice([0.0, 1.0, 2.0], size=n, p=[0.92, 0.06, 0.02])
        b = (cat == 0.0).astype(float)                  # complete separation
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        X = np.column_stack([b, cat, x1, x2])
        block = rng.random(n) < 0.25                    # shared missing block
        X[block, 0] = np.nan
        X[block, 1] = np.nan
        return MICEDesign.from_array(
            X, column_kinds=["binary", "categorical", "numeric", "numeric"]
        )

    def test_completes_and_does_not_collapse(self):
        d = self._separated_design()
        m = 20
        sol = mice(d, n_imputations=m, max_iter=10, seed=13, backend="gpu")
        assert sol.info["device"] == "mps"
        imp = sol.imputations(0)                         # (m, n_mis)
        for dset in sol.completed_datasets():
            assert np.isfinite(dset).all()
        # POOLED majority proportion must not collapse toward the minority
        # class (the 6.0.0 silent failure imputed ~0.29 pooled on GSS's
        # analogue; a healthy run sits ~0.75-0.85). Deliberately NOT a
        # per-chain bound: on a doubly-missing deterministic pair the
        # R-style posterior draw under separation has sd ~ 1/sqrt(lam) along
        # the separated direction, so individual chains legitimately disperse
        # (and occasionally invert the coupling) on the CPU reference too —
        # measured CPU per-chain majority 0.18-0.95 at this config. The CPU
        # is the calibration oracle (companion test), not per-chain
        # realizations.
        assert imp.mean() > 0.6, f"pooled collapse: {imp.mean():.3f}"

    def test_matches_cpu_under_separation(self):
        d = self._separated_design(seed=1)
        m = 20
        cpu = mice(d, n_imputations=m, max_iter=10, seed=7, backend="cpu")
        gpu = mice(d, n_imputations=m, max_iter=10, seed=7, backend="gpu")
        c1 = cpu.imputations(0).ravel().mean()
        g1 = gpu.imputations(0).ravel().mean()
        assert abs(c1 - g1) < 0.06, f"separated binary drift: cpu={c1:.3f} gpu={g1:.3f}"


class TestMpsPolyregCompleteSeparation:
    """A completely separated UNORDERED target must fit finite on MPS fp32 —
    the polyreg sibling of the GSS logreg refusal (6.1.x).

    The pre-fix ``_gpu_polyreg`` carried the same defect as the pre-fix
    logreg: Hessian-only ridge, unpenalised score. Under complete separation
    the fp32 batched Newton saturated at |beta| ~1e26 FINITE, and the
    penalty-free posterior draw collapsed every missing cell of every chain
    onto ONE category — silent corruption the end-of-sweep non-finite guard
    cannot see (measured: imputed marginal [0, 0, 1] on a 92/6/2 column).
    With the CPU slope ridge ported (in the score, intercepts free) plus the
    damped step, the fit shrinks to the CPU optimum and the imputed marginal
    tracks the CPU reference.
    """

    @staticmethod
    def _separated_design(seed=0):
        # Mirror the GSS structure with an unordered pair: two 3-level
        # categorical columns that are identical by construction (an identity
        # recode), imbalanced 92/6/2, missing in the SAME rows — so every
        # chained-equations fit of either column on the other's dummies is
        # COMPLETELY separated in every class direction.
        rng = np.random.default_rng(seed)
        n = 2000
        cat = rng.choice([0.0, 1.0, 2.0], size=n, p=[0.92, 0.06, 0.02])
        X = np.column_stack(
            [cat, cat.copy(), rng.standard_normal(n), rng.standard_normal(n)]
        )
        block = rng.random(n) < 0.25                    # shared missing block
        X[block, 0] = np.nan
        X[block, 1] = np.nan
        return MICEDesign.from_array(
            X, column_kinds=["categorical", "categorical", "numeric", "numeric"]
        )

    def test_completes_and_no_chain_collapses(self):
        d = self._separated_design()
        m = 20
        sol = mice(d, n_imputations=m, max_iter=10, seed=13, backend="gpu")
        assert sol.info["device"] == "mps"
        for dset in sol.completed_datasets():
            assert np.isfinite(dset).all()
        imp = sol.imputations(0)                         # (m, n_mis)
        # The pre-fix failure imputes ONE constant category in every chain.
        constant_chains = sum(np.unique(imp[c]).size < 2 for c in range(m))
        assert constant_chains < m // 2, (
            f"{constant_chains}/{m} chains imputed a constant category"
        )
        # All three classes must appear somewhere, and the pooled majority
        # share must stay majority (the collapse landed on a minority class).
        assert np.unique(imp).size == 3
        assert (imp == 0.0).mean() > 0.5

    def test_matches_cpu_under_separation(self):
        d = self._separated_design(seed=1)
        m = 20
        cpu = mice(d, n_imputations=m, max_iter=10, seed=7, backend="cpu")
        gpu = mice(d, n_imputations=m, max_iter=10, seed=7, backend="gpu")
        levels = [0.0, 1.0, 2.0]
        ci, gi = cpu.imputations(0).ravel(), gpu.imputations(0).ravel()
        cp = np.array([np.mean(ci == lv) for lv in levels])
        gp = np.array([np.mean(gi == lv) for lv in levels])
        tv = 0.5 * float(np.abs(cp - gp).sum())
        assert tv < 0.10, f"separated nominal drift: cpu={np.round(cp, 3)} gpu={np.round(gp, 3)}"
