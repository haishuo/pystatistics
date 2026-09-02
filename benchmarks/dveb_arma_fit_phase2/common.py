"""Frozen identities and parameter mappings for exact-ARMA Phase II."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
PHASE0_PATH = ROOT / "benchmarks/dveb_arma_phase0b/common.py"


def _load_phase0():
    name = "_pystatistics_dveb_arma_phase0b_common"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PHASE0_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen Phase-0B identities from {PHASE0_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PHASE0 = _load_phase0()
CELLS = PHASE0.CELLS
FAMILIES = PHASE0.FAMILIES
generate_cell = PHASE0.generate_cell
sha256_array = PHASE0.sha256_array


@dataclass(frozen=True)
class FitFamily:
    family_id: str
    ar_free: tuple[int, ...]
    ma_free: tuple[int, ...]

    @property
    def state(self) -> int:
        return FAMILIES[self.family_id].r

    @property
    def parameter_count(self) -> int:
        return len(self.ar_free) + len(self.ma_free)


FIT_FAMILIES = {
    "F1": FitFamily("F1", (0,), ()),
    "F3": FitFamily("F3", (0, 1, 2), (0, 1)),
    "F13": FitFamily("F13", (0, 11, 12), (0,)),
    "F25": FitFamily("F25", (0, 23, 24), (0,)),
}

CALIBRATION_CELLS = ("C01", "C02", "C03", "C04")
EVALUATION_CELLS = tuple(f"E{index:02d}" for index in range(1, 13))
PRIMARY_CELLS = ("E04", "E05", "E06", "E07", "E08", "E09", "E11", "E12")
SECONDARY_CELLS = ("E01", "E02", "E03", "E10")
MAX_ACTIVE_OPTIMIZERS = 256
OPTIMIZER_WORKERS = 12
THREAD_STACK_BYTES = 2 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def family_for_cell(cell_id: str) -> FitFamily:
    return FIT_FAMILIES[CELLS[cell_id].family_id]


def truth_vector(family_id: str) -> NDArray[np.float64]:
    spec = FIT_FAMILIES[family_id]
    family = FAMILIES[family_id]
    values = [family.phi[index] for index in spec.ar_free]
    values.extend(family.noise_loading[index + 1] for index in spec.ma_free)
    return np.asarray(values, dtype=np.float64)


def expand_parameters(
    parameters: NDArray[np.float64], family_id: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Expand free rows into full AR and ``[1, MA...]`` loading arrays."""
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    spec = FIT_FAMILIES[family_id]
    if values.ndim != 2 or values.shape[1] != spec.parameter_count:
        raise ValueError(
            f"{family_id} expects (*,{spec.parameter_count}) free parameters, "
            f"got {values.shape}"
        )
    phi = np.zeros((values.shape[0], spec.state), dtype=np.float64)
    loading = np.zeros_like(phi)
    loading[:, 0] = 1.0
    split = len(spec.ar_free)
    for column, lag in enumerate(spec.ar_free):
        phi[:, lag] = values[:, column]
    for column, lag in enumerate(spec.ma_free, start=split):
        loading[:, lag + 1] = values[:, column]
    return np.ascontiguousarray(phi), np.ascontiguousarray(loading)


def minimum_ar_root(parameters: NDArray[np.float64], family_id: str) -> float:
    phi, _ = expand_parameters(np.asarray(parameters), family_id)
    coefficients = np.concatenate((-phi[0, ::-1], np.ones(1, dtype=np.float64)))
    return float(np.min(np.abs(np.roots(coefficients))))


def biased_yule_walker(y: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    """The existing single-series biased Yule--Walker start, kept byte-direct."""
    y = np.asarray(y, dtype=np.float64)
    if order == 0:
        return np.empty(0, dtype=np.float64)
    n = y.size
    acov = np.empty(order + 1, dtype=np.float64)
    for lag in range(order + 1):
        acov[lag] = np.dot(y[: n - lag], y[lag:]) / n
    if acov[0] <= 0.0:
        return np.zeros(order, dtype=np.float64)
    toeplitz = np.empty((order, order), dtype=np.float64)
    for row in range(order):
        for column in range(order):
            toeplitz[row, column] = acov[abs(row - column)]
    try:
        return np.linalg.solve(toeplitz, acov[1 : order + 1])
    except np.linalg.LinAlgError:
        return np.zeros(order, dtype=np.float64)


def generate_starts(cell_id: str) -> NDArray[np.float64]:
    """Generate the protocol's deterministic stationary free-parameter starts."""
    z, _, _ = generate_cell(cell_id)
    spec = family_for_cell(cell_id)
    starts = np.zeros((z.shape[0], spec.parameter_count), dtype=np.float64)
    for row in range(z.shape[0]):
        full_ar = np.clip(biased_yule_walker(z[row], spec.state), -0.99, 0.99)
        starts[row, : len(spec.ar_free)] = full_ar[list(spec.ar_free)]
        for _ in range(65):
            if minimum_ar_root(starts[row], spec.family_id) >= 1.001:
                break
            starts[row, : len(spec.ar_free)] *= 0.5
        else:
            raise RuntimeError(f"{cell_id} row {row}: start stationarity repair exhausted")
    return np.ascontiguousarray(starts)


def public_airpassengers() -> NDArray[np.float64]:
    fixture = ROOT / "tests/fixtures/arima_kalman_r_reference.json"
    values = json.loads(fixture.read_text())["series"]["airpassengers"]
    differenced = np.diff(np.log(np.asarray(values, dtype=np.float64)))
    differenced -= np.mean(differenced)
    return np.ascontiguousarray(differenced, dtype=np.float64)


def input_start_record(cell_id: str) -> dict[str, object]:
    cell = CELLS[cell_id]
    family = family_for_cell(cell_id)
    z, truth_phi, truth_loading = generate_cell(cell_id)
    starts = generate_starts(cell_id)
    start_phi, start_loading = expand_parameters(starts, family.family_id)
    roots = np.asarray(
        [minimum_ar_root(row, family.family_id) for row in starts], dtype=np.float64
    )
    fixed_ar = tuple(index for index in range(family.state) if index not in family.ar_free)
    fixed_ma = tuple(
        index for index in range(max(0, family.state - 1)) if index not in family.ma_free
    )
    return {
        "cell_id": cell_id,
        "phase": cell.phase,
        "primary": cell.primary,
        "shape": {"k": cell.k, "n": cell.n, "state": family.state},
        "family": family.family_id,
        "free_ar_zero_based": list(family.ar_free),
        "free_ma_zero_based": list(family.ma_free),
        "fixed_ar_zero_based": list(fixed_ar),
        "fixed_ma_zero_based": list(fixed_ma),
        "minimum_start_ar_root_modulus": float(roots.min()),
        "sha256": {
            "z": sha256_array(z),
            "truth_phi": sha256_array(truth_phi),
            "truth_loading": sha256_array(truth_loading),
            "starts": sha256_array(starts),
            "start_phi": sha256_array(start_phi),
            "start_loading": sha256_array(start_loading),
        },
    }
