"""NumPy and Cython authorities for fixed-parameter exact ARMA batches."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from pystatistics.timeseries._arima_kalman_kernel import ArmaKalmanWorkspace

from pystatistics.timeseries._arima_kalman_ref import kalman_loop, stationary_init

KAPPA = 1.0e6
PENALTY = 1.0e18


def _finish(n: int, ok: bool, sse: float, sum_log_f: float) -> tuple[float, float, bool]:
    if not ok or not math.isfinite(sse) or sse <= 0.0:
        return PENALTY, 1.0, False
    sigma2 = sse / n
    nll = 0.5 * n * math.log(2.0 * math.pi * sigma2) + 0.5 * sum_log_f + 0.5 * n
    if not math.isfinite(nll):
        return PENALTY, 1.0, False
    return nll, sigma2, True


def numpy_one(z: NDArray, phi: NDArray, loading: NDArray) -> tuple[float, float, bool]:
    r = phi.size
    p, init_ok = stationary_init(phi, loading)
    if not init_ok:
        p = KAPPA * np.eye(r, dtype=np.float64)
    a = np.zeros(r, dtype=np.float64)
    innov, f, ok = kalman_loop(z, phi, loading, a, p)
    if not ok:
        return PENALTY, 1.0, False
    with np.errstate(divide="ignore", invalid="ignore"):
        sse = float(np.sum(innov * innov / f))
        sum_log_f = float(np.sum(np.log(f)))
    return _finish(z.size, ok, sse, sum_log_f)


def cython_batch(z: NDArray, phi: NDArray, loading: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    k, n = z.shape
    r = phi.shape[1]
    nll = np.empty(k, dtype=np.float64)
    sigma2 = np.empty(k, dtype=np.float64)
    status = np.empty(k, dtype=np.bool_)
    workspaces = [ArmaKalmanWorkspace(r, n) for _ in range(k)]
    for row in range(k):
        ok, sse, sum_log_f = workspaces[row].loglik_parts(z[row], phi[row], loading[row], KAPPA)
        nll[row], sigma2[row], status[row] = _finish(n, ok, sse, sum_log_f)
    return nll, sigma2, status


def numpy_sample(
    z: NDArray, phi: NDArray, loading: NDArray, rows: int = 4
) -> tuple[NDArray, NDArray, NDArray]:
    count = min(rows, z.shape[0])
    answers = [numpy_one(z[i], phi[i], loading[i]) for i in range(count)]
    return (
        np.asarray([item[0] for item in answers], dtype=np.float64),
        np.asarray([item[1] for item in answers], dtype=np.float64),
        np.asarray([item[2] for item in answers], dtype=np.bool_),
    )
