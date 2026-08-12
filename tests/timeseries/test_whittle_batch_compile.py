"""Tests for the batched Whittle NLL's default torch.compile path.

Pins the three guarantees of the compiled-by-default design:
(1) on a GPU device with torch.compile available, the NLL runs compiled and
    the Solution discloses it (``info['nll_compiled']``);
(2) compiled and eager produce the same estimates at the GPU_FP32 tier;
(3) if the compile machinery fails on first evaluation, the fit falls back
    to the eager path with a RuntimeWarning, still succeeds, and disclosure
    reports the eager mode — while a genuine numerical error still propagates.
"""

import warnings

import numpy as np
import pytest

from pystatistics.core.compute.tolerances import GPU_FP32


def _gpu_device():
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return None


pytestmark = pytest.mark.skipif(
    _gpu_device() is None, reason="no torch GPU (CUDA/MPS)"
)


def _arma_series(K=64, n=1200, seed=0):
    rng = np.random.default_rng(seed)
    Y = np.zeros((K, n))
    for k in range(K):
        e = rng.standard_normal(n + 100)
        y = np.zeros(n + 100)
        for t in range(2, n + 100):
            y[t] = 0.5 * y[t - 1] - 0.2 * y[t - 2] + e[t] + 0.3 * e[t - 1]
        Y[k] = y[100:]
    return Y


def _fit_eager(Y):
    """Fit with the compile path disabled (pending flag cleared)."""
    import pystatistics.timeseries.backends.whittle_batch_gpu as wb

    orig_init = wb.BatchedWhittleGPU.__init__

    def eager_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._nll = self._nll_eager
        self._nll_pending_first_call = False

    wb.BatchedWhittleGPU.__init__ = eager_init
    try:
        from pystatistics.timeseries import arima_batch

        return arima_batch(Y, order=(2, 0, 1), backend="gpu")
    finally:
        wb.BatchedWhittleGPU.__init__ = orig_init


class TestWhittleBatchCompile:
    def test_compiled_by_default_and_disclosed(self):
        from pystatistics.timeseries import arima_batch

        Y = _arma_series()
        sol = arima_batch(Y, order=(2, 0, 1), backend="gpu")
        assert sol.info["nll_compiled"] is True

    def test_compiled_matches_eager_at_tier(self):
        from pystatistics.timeseries import arima_batch

        Y = _arma_series()
        sol_c = arima_batch(Y, order=(2, 0, 1), backend="gpu")
        sol_e = _fit_eager(Y)
        assert sol_e.info["nll_compiled"] is False
        for attr in ("ar", "ma", "sigma2"):
            a = np.asarray(getattr(sol_e, attr), dtype=float)
            b = np.asarray(getattr(sol_c, attr), dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            assert ok.any()
            np.testing.assert_allclose(
                a[ok], b[ok], rtol=GPU_FP32.rtol, atol=GPU_FP32.atol
            )

    def test_compile_failure_falls_back_with_warning(self):
        import pystatistics.timeseries.backends.whittle_batch_gpu as wb
        from pystatistics.timeseries import arima_batch

        orig_init = wb.BatchedWhittleGPU.__init__

        def broken_compile_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)

            def boom(params_batch):
                raise RuntimeError("simulated inductor failure")

            self._nll = boom
            self._nll_pending_first_call = True

        wb.BatchedWhittleGPU.__init__ = broken_compile_init
        try:
            Y = _arma_series(K=16)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sol = arima_batch(Y, order=(2, 0, 1), backend="gpu")
            msgs = [str(w.message) for w in caught if w.category is RuntimeWarning]
            assert any("eager path" in m for m in msgs)
            assert sol.info["nll_compiled"] is False
            assert np.isfinite(np.asarray(sol.sigma2, dtype=float)).all()
        finally:
            wb.BatchedWhittleGPU.__init__ = orig_init

    def test_genuine_error_still_propagates(self):
        import torch

        import pystatistics.timeseries.backends.whittle_batch_gpu as wb

        Y = _arma_series(K=8)
        fitter = wb.BatchedWhittleGPU(
            Y - Y.mean(axis=1, keepdims=True), 2, 1, device=_gpu_device()
        )
        # A shape-invalid params batch must raise from BOTH paths — the
        # first-call fallback must not swallow a real error.
        bad = torch.zeros((3, 1), device=fitter._device, dtype=fitter._dtype)
        with pytest.raises(Exception):
            fitter._eval_nll(bad)
