"""Frozen inputs and reference helpers for the exact-ARMA Phase-0B screen."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MASTER_SEED = 0x445645425F41524D
BURN_IN = 512


@dataclass(frozen=True)
class Family:
    family_id: str
    phi: tuple[float, ...]
    noise_loading: tuple[float, ...]

    @property
    def r(self) -> int:
        return len(self.phi)


@dataclass(frozen=True)
class Cell:
    cell_id: str
    k: int
    n: int
    family_id: str
    phase: str
    primary: bool = False


def _sparse(length: int, values: dict[int, float]) -> tuple[float, ...]:
    result = [0.0] * length
    for one_based_lag, value in values.items():
        result[one_based_lag - 1] = value
    return tuple(result)


FAMILIES = {
    "F1": Family("F1", (0.60,), (1.0,)),
    "F3": Family("F3", (0.50, -0.20, 0.05), (1.0, 0.30, -0.10)),
    # (1 - 0.4 B)(1 - 0.45 B^12): phi_1=.4, phi_12=.45,
    # phi_13=-.18. The corresponding seasonal-root modulus is > 1.05.
    "F13": Family(
        "F13",
        _sparse(13, {1: 0.40, 12: 0.45, 13: -0.18}),
        _sparse(13, {1: 1.0, 2: 0.20}),
    ),
    # (1 - 0.25 B)(1 - 0.20 B^24): phi_1=.25, phi_24=.20,
    # phi_25=-.05. The seasonal-root modulus is approximately 1.069.
    "F25": Family(
        "F25",
        _sparse(25, {1: 0.25, 24: 0.20, 25: -0.05}),
        _sparse(25, {1: 1.0, 2: -0.25}),
    ),
}


CELLS = {
    "C01": Cell("C01", 8, 120, "F1", "calibration"),
    "C02": Cell("C02", 64, 500, "F3", "calibration"),
    "C03": Cell("C03", 256, 2_000, "F13", "calibration"),
    "C04": Cell("C04", 1_024, 500, "F25", "calibration"),
    "E01": Cell("E01", 1, 120, "F1", "evaluation"),
    "E02": Cell("E02", 1, 2_000, "F13", "evaluation"),
    "E03": Cell("E03", 16, 500, "F3", "evaluation"),
    "E04": Cell("E04", 64, 2_000, "F3", "evaluation", True),
    "E05": Cell("E05", 256, 500, "F13", "evaluation", True),
    "E06": Cell("E06", 256, 2_000, "F13", "evaluation", True),
    "E07": Cell("E07", 1_024, 2_000, "F3", "evaluation", True),
    "E08": Cell("E08", 1_024, 500, "F13", "evaluation", True),
    "E09": Cell("E09", 64, 10_000, "F3", "evaluation", True),
    "E10": Cell("E10", 16, 2_000, "F25", "evaluation"),
    "E11": Cell("E11", 256, 500, "F25", "evaluation", True),
    "E12": Cell("E12", 1_024, 120, "F1", "evaluation", True),
}


def sha256_array(value: NDArray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def ar_root_moduli(phi: tuple[float, ...]) -> NDArray[np.float64]:
    coefficients = np.concatenate(
        (-np.asarray(phi, dtype=np.float64)[::-1], np.ones(1, dtype=np.float64))
    )
    return np.abs(np.roots(coefficients))


def cell_seed(cell_id: str) -> int:
    digest = hashlib.sha256(f"{MASTER_SEED}:{cell_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def generate_cell(cell_id: str) -> tuple[NDArray, NDArray, NDArray]:
    """Return byte-stable `(z, phi, R)` for one frozen cell."""
    cell = CELLS[cell_id]
    family = FAMILIES[cell.family_id]
    phi_one = np.asarray(family.phi, dtype=np.float64)
    loading_one = np.asarray(family.noise_loading, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64DXSM(cell_seed(cell_id)))
    innovations = rng.standard_normal((cell.k, cell.n + BURN_IN + family.r))
    y = np.zeros_like(innovations)

    for t in range(family.r, innovations.shape[1]):
        ar_value = np.zeros(cell.k, dtype=np.float64)
        ma_value = innovations[:, t].copy()
        for lag, coefficient in enumerate(phi_one, start=1):
            ar_value += coefficient * y[:, t - lag]
        for lag, coefficient in enumerate(loading_one[1:], start=1):
            ma_value += coefficient * innovations[:, t - lag]
        y[:, t] = ar_value + ma_value

    z = np.ascontiguousarray(y[:, -cell.n :], dtype=np.float64)
    # ``ascontiguousarray`` may preserve the read-only flag of a broadcast view
    # when K=1 because no layout copy is necessary.  The consumer contract is
    # caller-owned, writable C storage at every K, so force an owning copy.
    phi = np.array(np.broadcast_to(phi_one, (cell.k, family.r)), copy=True, order="C")
    loading = np.array(np.broadcast_to(loading_one, (cell.k, family.r)), copy=True, order="C")
    return z, phi, loading


def input_record(cell_id: str) -> dict[str, object]:
    cell = CELLS[cell_id]
    family = FAMILIES[cell.family_id]
    z, phi, loading = generate_cell(cell_id)
    roots = ar_root_moduli(family.phi)
    return {
        "cell_id": cell_id,
        "phase": cell.phase,
        "primary": cell.primary,
        "shape": {"k": cell.k, "n": cell.n, "r": family.r},
        "family": family.family_id,
        "seed": cell_seed(cell_id),
        "sha256": {
            "z": sha256_array(z),
            "phi": sha256_array(phi),
            "noise_loading": sha256_array(loading),
        },
        "minimum_ar_root_modulus": float(roots.min()),
    }
