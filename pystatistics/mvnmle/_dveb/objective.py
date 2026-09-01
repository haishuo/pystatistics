"""PyStatistics-owned objective adapter for the DVEB dense CPU ABI."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pystatistics.core.exceptions import DimensionError, ValidationError
from pystatistics.mvnmle._objectives._batched_cholesky import build_batched_constants
from pystatistics.mvnmle._objectives.base import MLEObjectiveBase
from pystatistics.mvnmle._objectives.parameterizations import CholeskyParameterization

from .loader import (
    LIBRARY_SHA256,
    SCHEDULE_AUTO,
    VALID_SCHEDULES,
    DenseContext,
    DenseLibrary,
)


def _affinity_threads() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _as_i64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def _as_f64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


class DVEBDenseObjective(MLEObjectiveBase):
    """Forward-Cholesky MVN objective evaluated by the qualified DVEB artifact."""

    epsilon = 1e-12

    def __init__(
        self,
        data: np.ndarray,
        *,
        validate: bool = True,
        threads: int | None = None,
        schedule: int = SCHEDULE_AUTO,
        library_path: str | Path | None = None,
    ):
        super().__init__(data, skip_validation=not validate)
        self.parameterization = CholeskyParameterization(self.n_vars)
        self.n_params = self.parameterization.n_params
        self.threads = _affinity_threads() if threads is None else int(threads)
        if self.threads < 1:
            raise ValidationError(f"DVEB threads must be positive, got {self.threads}")
        if schedule not in VALID_SCHEDULES:
            raise ValidationError(f"Unknown DVEB schedule {schedule}")
        self.schedule = int(schedule)
        self.last_selected_schedule: int | None = None

        arrays = self._pack_context_arrays()
        self._library = DenseLibrary(library_path)
        self._context = DenseContext(
            self._library,
            p=self.n_vars,
            patterns=self.n_patterns,
            epsilon=self.epsilon,
            max_threads=self.threads,
            **arrays,
        )
        self._scratch_bytes = self._context.scratch_bytes

    def _pack_context_arrays(self) -> dict[str, np.ndarray]:
        constants = build_batched_constants(
            self.patterns,
            self.n_vars,
            var_scale=self.jitter_scale(),
        )
        lengths = np.fromiter(
            (len(pattern.observed_indices) for pattern in self.patterns),
            dtype=np.int64,
            count=self.n_patterns,
        )
        offsets = np.empty(self.n_patterns + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        centered_offsets = np.empty(self.n_patterns + 1, dtype=np.int64)
        centered_offsets[0] = 0
        np.cumsum(lengths * lengths, out=centered_offsets[1:])

        observed_parts = []
        ybar_parts = []
        centered_parts = []
        jitter_parts = []
        for k, length in enumerate(lengths):
            v = int(length)
            observed_parts.append(constants.obs_idx[k, :v])
            ybar_parts.append(constants.ybar[k, :v])
            centered_parts.append(constants.c[k, :v, :v].reshape(-1))
            jitter_parts.append(constants.diag_scale[k, :v])

        return {
            "offsets": _as_i64(offsets),
            "centered_offsets": _as_i64(centered_offsets),
            "observed_index": _as_i64(np.concatenate(observed_parts)),
            "n_k": _as_f64(constants.n_k),
            "ybar": _as_f64(np.concatenate(ybar_parts)),
            "centered": _as_f64(np.concatenate(centered_parts)),
            "jitter_scale": _as_f64(np.concatenate(jitter_parts)),
        }

    @property
    def scratch_bytes(self) -> int:
        return self._scratch_bytes

    @property
    def artifact_sha256(self) -> str:
        return LIBRARY_SHA256

    def get_initial_parameters(self) -> np.ndarray:
        return self.parameterization.get_initial_parameters(
            self.sample_mean,
            self.sample_cov,
        )

    def _validate_theta(self, theta) -> np.ndarray:
        if not isinstance(theta, np.ndarray):
            raise ValidationError("DVEB theta must be a NumPy array; no implicit copy is made")
        if theta.dtype != np.float64:
            raise ValidationError(f"DVEB theta must have dtype float64, got {theta.dtype}")
        if theta.ndim != 1 or theta.shape != (self.n_params,):
            raise DimensionError(
                f"DVEB theta must have shape ({self.n_params},), got {theta.shape}"
            )
        if not theta.flags.c_contiguous:
            raise ValidationError("DVEB theta must be C-contiguous; no implicit copy is made")
        if not np.all(np.isfinite(theta)):
            raise ValidationError("DVEB theta must contain only finite values")
        return theta

    def compute_value_and_gradient(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        theta = self._validate_theta(theta)
        gradient = np.empty(self.n_params, dtype=np.float64)
        value, selected = self._context.value_gradient(
            theta,
            gradient,
            threads=self.threads,
            schedule=self.schedule,
        )
        self.last_selected_schedule = selected
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise RuntimeError("DVEB returned non-finite output after reporting success")
        return value, gradient

    def compute_objective(self, theta: np.ndarray) -> float:
        value, _ = self.compute_value_and_gradient(theta)
        return value

    def compute_gradient(self, theta: np.ndarray) -> np.ndarray:
        _, gradient = self.compute_value_and_gradient(theta)
        return gradient

    def extract_parameters(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        theta = self._validate_theta(theta)
        mu, sigma = self.parameterization.unpack(theta)
        neg2_loglik = self.compute_objective(theta)
        return mu, sigma, -0.5 * neg2_loglik

    def close(self) -> None:
        self._context.close()

    def clear_cache(self) -> None:
        self.close()
