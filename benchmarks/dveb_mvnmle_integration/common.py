"""Frozen cases and result checks for the DVEB MVN-MLE integration campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

THREADS = (1, 6, 12)
CASE_IDS = ("E1", "E2", "E3", "E4", "E5", "E6")
SEED = 0xD0EB1008
WARMUPS = 3
REPETITIONS = 30


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    n: int
    p: int
    missingness: str
    seed: int


CASES = {
    "E1": CaseSpec("E1", 150, 4, "mcar15", 0),
    "E2": CaseSpec("E2", 178, 13, "mcar15", 1),
    "E3": CaseSpec("E3", 569, 30, "mcar15", 2),
    "E4": CaseSpec("E4", 5000, 13, "structured32", 3),
    "E5": CaseSpec("E5", 5000, 30, "structured64", 4),
    "E6": CaseSpec("E6", 5000, 30, "mcar15", 5),
}


def _repair_mask(mask: np.ndarray) -> np.ndarray:
    """Ensure every row and column contains at least one observed value."""
    mask = mask.copy()
    n, p = mask.shape
    for row in np.flatnonzero(~mask.any(axis=1)):
        mask[row, row % p] = True
    for column in np.flatnonzero(~mask.any(axis=0)):
        mask[column % n, column] = True
    return mask


def make_case(case_id: str) -> np.ndarray:
    """Reproduce the prospectively frozen Trunk 008 evaluation case."""
    spec = CASES[case_id]
    rng = np.random.default_rng(spec.seed)
    indices = np.arange(spec.p)
    covariance = 0.35 ** np.abs(indices[:, None] - indices[None, :])
    mean = np.linspace(-0.5, 0.5, spec.p)
    data = rng.multivariate_normal(mean, covariance, size=spec.n).astype(np.float64)

    if spec.missingness.startswith("mcar"):
        rate = float(spec.missingness.removeprefix("mcar")) / 100.0
        observed = rng.random(data.shape) >= rate
    else:
        pattern_count = int(spec.missingness.removeprefix("structured"))
        pattern_library = rng.random((pattern_count, spec.p)) >= 0.15
        pattern_library = _repair_mask(pattern_library)
        observed = pattern_library[rng.integers(0, pattern_count, size=spec.n)]

    observed = _repair_mask(observed)
    data[~observed] = np.nan
    return np.ascontiguousarray(data)


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def solution_record(solution: Any) -> dict[str, Any]:
    mu = np.ascontiguousarray(solution.muhat, dtype=np.float64)
    sigma = np.ascontiguousarray(solution.sigmahat, dtype=np.float64)
    packed = np.concatenate((np.asarray([solution.loglik]), mu, sigma.reshape(-1)))
    return {
        "backend": solution.backend_name,
        "converged": bool(solution.converged),
        "n_iter": int(solution.n_iter),
        "loglik": float(solution.loglik),
        "gradient_norm": (
            None if solution.gradient_norm is None else float(solution.gradient_norm)
        ),
        "muhat": mu.tolist(),
        "sigmahat": sigma.tolist(),
        "checksum": array_sha256(packed),
        "info": _jsonable(solution.info),
        "timing": _jsonable(solution.timing or {}),
        "warnings": list(solution.warnings),
    }


def compare_solutions(torch_record: dict[str, Any], dveb_record: dict[str, Any]) -> dict[str, Any]:
    torch_mu = np.asarray(torch_record["muhat"], dtype=np.float64)
    dveb_mu = np.asarray(dveb_record["muhat"], dtype=np.float64)
    torch_sigma = np.asarray(torch_record["sigmahat"], dtype=np.float64)
    dveb_sigma = np.asarray(dveb_record["sigmahat"], dtype=np.float64)
    loglik_abs = abs(torch_record["loglik"] - dveb_record["loglik"])
    mu_max_abs = float(np.max(np.abs(torch_mu - dveb_mu)))
    sigma_max_abs = float(np.max(np.abs(torch_sigma - dveb_sigma)))
    mu_close = bool(np.allclose(torch_mu, dveb_mu, rtol=1.0e-3, atol=1.0e-3))
    sigma_close = bool(np.allclose(torch_sigma, dveb_sigma, rtol=1.0e-3, atol=1.0e-3))
    passed = bool(
        torch_record["converged"]
        and dveb_record["converged"]
        and loglik_abs <= 1.0e-5
        and mu_close
        and sigma_close
    )
    return {
        "pass": passed,
        "loglik_abs": loglik_abs,
        "loglik_tolerance": 1.0e-5,
        "mu_max_abs": mu_max_abs,
        "sigma_max_abs": sigma_max_abs,
        "parameter_rtol": 1.0e-3,
        "parameter_atol": 1.0e-3,
        "mu_close": mu_close,
        "sigma_close": sigma_close,
        "both_converged": bool(torch_record["converged"] and dveb_record["converged"]),
    }


def affinity() -> list[int]:
    import os

    return sorted(os.sched_getaffinity(0))


def configure_threads(count: int) -> None:
    import torch

    torch.set_num_threads(count)
    torch.set_num_interop_threads(1)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
