"""Complete private exact-ARMA fits for the frozen Phase-II study."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from pystatistics.timeseries._arima_factored import normalize_ma_coefficients
from pystatistics.timeseries._arima_kalman_kernel import ArmaKalmanWorkspace

from benchmarks.dveb_arma_phase0b.reference_impl import KAPPA, _finish

from .backends import expand_fit_parameters, fit_layout
from .coordinator import CoordinatorAccounting, DeterministicCoordinator, OPTIMIZER_OPTIONS


@dataclass(frozen=True)
class FitRow:
    row: int
    success: bool
    parameters: tuple[float, ...]
    nll: float
    sigma2: float
    reported_nll: float
    reported_sigma2: float
    reported_status: bool
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    callback_count: int
    final_gradient_norm: float


@dataclass(frozen=True)
class FitBatch:
    route: str
    family_id: str
    rows: tuple[FitRow, ...]
    coordinator: CoordinatorAccounting | None

    @property
    def parameters(self) -> NDArray[np.float64]:
        return np.asarray([row.parameters for row in self.rows], dtype=np.float64)

    @property
    def nll(self) -> NDArray[np.float64]:
        return np.asarray([row.nll for row in self.rows], dtype=np.float64)

    @property
    def sigma2(self) -> NDArray[np.float64]:
        return np.asarray([row.sigma2 for row in self.rows], dtype=np.float64)

    @property
    def success(self) -> NDArray[np.bool_]:
        return np.asarray([row.success for row in self.rows], dtype=np.bool_)

    def record(self) -> dict:
        return {
            "route": self.route,
            "family_id": self.family_id,
            "rows": [asdict(row) for row in self.rows],
            "coordinator": asdict(self.coordinator) if self.coordinator else None,
        }


class CythonRowObjective:
    def __init__(self, z: NDArray[np.float64], family_id: str):
        self.z = np.ascontiguousarray(z, dtype=np.float64)
        self.family_id = family_id
        state, _ar, _ma = fit_layout(family_id)
        self.workspace = ArmaKalmanWorkspace(state, z.size)
        self.last_sigma2 = 1.0
        self.last_status = False

    def evaluate(self, parameters: NDArray[np.float64]) -> tuple[float, float, bool]:
        phi, loading = expand_fit_parameters(parameters, self.family_id)
        ok, sse, sum_log_f = self.workspace.loglik_parts(
            self.z, phi[0], loading[0], KAPPA
        )
        value, sigma2, status = _finish(self.z.size, ok, sse, sum_log_f)
        self.last_sigma2 = sigma2
        self.last_status = status
        return value, sigma2, status

    def __call__(self, parameters: NDArray[np.float64]) -> float:
        return self.evaluate(parameters)[0]


def _minimum_root(parameters: NDArray[np.float64], family_id: str) -> float:
    phi, _loading = expand_fit_parameters(parameters, family_id)
    polynomial = np.concatenate((-phi[0, ::-1], np.ones(1, dtype=np.float64)))
    return float(np.min(np.abs(np.roots(polynomial))))


def _canonical(parameters: NDArray[np.float64], family_id: str) -> NDArray[np.float64]:
    values = np.array(parameters, dtype=np.float64, copy=True)
    _state, ar_free, ma_free = fit_layout(family_id)
    if ma_free:
        start = len(ar_free)
        normalized, _flipped = normalize_ma_coefficients(values[start:])
        values[start:] = normalized
    return values


def _gradient_norm(result: OptimizeResult) -> float:
    gradient = getattr(result, "jac", None)
    if gradient is None:
        return math.nan
    values = np.asarray(gradient, dtype=np.float64)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _finalize_row(
    row: int,
    z: NDArray[np.float64],
    family_id: str,
    result: OptimizeResult,
    reported: tuple[float, float, bool] | None = None,
) -> FitRow:
    parameters = _canonical(np.asarray(result.x, dtype=np.float64), family_id)
    authority = CythonRowObjective(z, family_id)
    nll, sigma2, status = authority.evaluate(parameters)
    stationary = bool(np.isfinite(parameters).all() and _minimum_root(parameters, family_id) > 1.0)
    success = bool(result.success and status and stationary and np.isfinite(nll))
    if reported is None:
        reported = (float(result.fun), sigma2, status)
    return FitRow(
        row=row,
        success=success,
        parameters=tuple(map(float, parameters)),
        nll=float(nll),
        sigma2=float(sigma2),
        reported_nll=float(reported[0]),
        reported_sigma2=float(reported[1]),
        reported_status=bool(reported[2]),
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        function_evaluations=int(getattr(result, "nfev", 0)),
        gradient_evaluations=int(getattr(result, "njev", 0) or 0),
        callback_count=0,
        final_gradient_norm=_gradient_norm(result),
    )


def fit_s0(
    z: NDArray[np.float64], starts: NDArray[np.float64], family_id: str
) -> FitBatch:
    """Current Cython likelihood with independent finite-difference L-BFGS-B."""
    z = np.ascontiguousarray(z, dtype=np.float64)
    starts = np.ascontiguousarray(starts, dtype=np.float64)

    def one(row: int) -> FitRow:
        objective = CythonRowObjective(z[row], family_id)
        options = dict(OPTIMIZER_OPTIONS)
        options["eps"] = 1.0e-8
        result = minimize(
            objective, starts[row], method="L-BFGS-B", jac=None, options=options
        )
        return _finalize_row(row, z[row], family_id, result)

    with ThreadPoolExecutor(max_workers=min(12, z.shape[0])) as pool:
        rows = tuple(pool.map(one, range(z.shape[0])))
    return FitBatch("S0", family_id, rows, None)


def fit_coordinated(
    z: NDArray[np.float64],
    starts: NDArray[np.float64],
    family_id: str,
    route: str,
    backend,
) -> FitBatch:
    coordinator = DeterministicCoordinator(backend)
    results, accounting = coordinator.fit(starts)
    canonical = np.stack(
        [_canonical(np.asarray(result.x, dtype=np.float64), family_id) for result in results]
    )
    reported_nll, reported_sigma2, reported_status = backend.final_values(
        tuple(range(z.shape[0])), canonical
    )
    rows = tuple(
        _finalize_row(
            row, z[row], family_id, result,
            (reported_nll[row], reported_sigma2[row], reported_status[row]),
        )
        for row, result in enumerate(results)
    )
    return FitBatch(route, family_id, rows, accounting)
