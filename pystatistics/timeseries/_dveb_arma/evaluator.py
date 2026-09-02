"""PyStatistics-owned private adapters for the DVEB exact-ARMA artifact."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from pystatistics.core.exceptions import DimensionError, ValidationError

from .loader import (
    COMBINED_LIBRARY_SHA256,
    CPU_ITEM_PARALLEL,
    CPU_LIBRARY_SHA256,
    CPU_SERIAL,
    VALID_CPU_SCHEDULES,
    VALID_CUDA_BLOCKS,
    CPUContext,
    CUDAHostContext,
    RecurrenceLibrary,
    f64_pointer,
    i64_pointer,
)

_CUDA_BLOCK_BY_STATE = (0, 32, 0, 32) + (0,) * 9 + (256,) + (0,) * 11 + (256,)


def _matrix(name: str, value) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValidationError(f"DVEB {name} must be a NumPy array; no implicit copy is made")
    if value.dtype != np.float64:
        raise ValidationError(f"DVEB {name} must have dtype float64, got {value.dtype}")
    if value.ndim != 2:
        raise DimensionError(f"DVEB {name} must be two-dimensional, got {value.ndim}")
    if not value.flags.c_contiguous:
        raise ValidationError(f"DVEB {name} must be C-contiguous; no implicit copy is made")
    return value


def _inputs(z, phi, loading) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = _matrix("z", z)
    phi = _matrix("phi", phi)
    loading = _matrix("loading", loading)
    if z.shape[0] < 1 or z.shape[1] < 1:
        raise DimensionError(f"DVEB z must have positive shape, got {z.shape}")
    if phi.shape != loading.shape or phi.shape[0] != z.shape[0]:
        raise DimensionError(
            f"DVEB batch shapes disagree: z={z.shape}, phi={phi.shape}, "
            f"loading={loading.shape}"
        )
    if not 1 <= phi.shape[1] <= 25:
        raise DimensionError(f"DVEB state extent must be in [1, 25], got {phi.shape[1]}")
    return z, phi, loading


def _outputs(items: int):
    return (
        np.empty(items, dtype=np.float64),
        np.empty(items, dtype=np.float64),
        np.empty(items, dtype=np.int64),
    )


class DVEBCPUExactArma:
    """Persistent torch-free CPU evaluator with an explicit legal schedule."""

    artifact_sha256 = CPU_LIBRARY_SHA256

    def __init__(
        self, *, max_threads: int, schedule: int,
        library_path: str | Path | None = None,
    ):
        self.max_threads = int(max_threads)
        self.schedule = int(schedule)
        if self.max_threads < 1:
            raise ValidationError(f"DVEB max_threads must be positive, got {max_threads}")
        if self.schedule not in VALID_CPU_SCHEDULES:
            raise ValidationError(
                "DVEB CPU integration requires a forced serial or item-parallel schedule; "
                "the current automatic selector is not production-qualified"
            )
        self.library = RecurrenceLibrary(cuda=False, path=library_path)
        self.context = CPUContext(self.library, self.max_threads)
        self.last_selected_schedule: int | None = None

    @property
    def scratch_bytes(self) -> int:
        return self.context.scratch_bytes

    def evaluate(self, z, phi, loading, *, threads: int):
        self.context.require_open()
        z, phi, loading = _inputs(z, phi, loading)
        threads = int(threads)
        if threads < 1 or threads > self.max_threads:
            raise ValidationError(
                f"DVEB threads must be in [1, {self.max_threads}], got {threads}"
            )
        nll, sigma2, status = _outputs(z.shape[0])
        selected = ctypes.c_int(-1)
        code = self.library.cdll.dveb_recurrence_cpu_run(
            self.context.pointer,
            f64_pointer(z), z.shape[0], z.shape[1],
            f64_pointer(phi), phi.shape[0], phi.shape[1],
            f64_pointer(loading), loading.shape[0], loading.shape[1],
            f64_pointer(nll), nll.size, f64_pointer(sigma2), sigma2.size,
            i64_pointer(status), status.size,
            threads, self.schedule, ctypes.byref(selected),
        )
        self.library.raise_status(code, "CPU evaluation")
        self.last_selected_schedule = int(selected.value)
        return nll, sigma2, status.astype(np.bool_)

    def close(self) -> None:
        self.context.close()

    def __enter__(self):
        self.context.require_open()
        return self

    def __exit__(self, *_args):
        self.close()


class DVEBCudaTransferExactArma:
    """Explicit transfer-inclusive CUDA evaluator; never falls back to CPU."""

    artifact_sha256 = COMBINED_LIBRARY_SHA256

    def __init__(
        self, *, max_items: int, max_steps: int, max_state: int,
        library_path: str | Path, device: int = 0,
    ):
        self.max_items = int(max_items)
        self.max_steps = int(max_steps)
        self.max_state = int(max_state)
        if self.max_items < 1 or self.max_steps < 1 or not 1 <= self.max_state <= 25:
            raise ValidationError("DVEB CUDA capacities must be positive with state in [1, 25]")
        self.library = RecurrenceLibrary(cuda=True, path=library_path)
        max_f64 = (
            self.max_items * self.max_steps
            + 2 * self.max_items * self.max_state
            + 2 * self.max_items
        )
        self.context = CUDAHostContext(
            self.library, max_f64=max_f64, max_i64=self.max_items,
            block_by_state=_CUDA_BLOCK_BY_STATE, device=int(device),
        )
        self.last_selected_block: int | None = None

    @property
    def payload_bytes(self) -> int:
        return self.context.payload_bytes

    def evaluate(self, z, phi, loading, *, block: int = 0):
        self.context.require_open()
        z, phi, loading = _inputs(z, phi, loading)
        if (z.shape[0] > self.max_items or z.shape[1] > self.max_steps
                or phi.shape[1] > self.max_state):
            raise DimensionError("DVEB CUDA input exceeds the declared context capacity")
        block = int(block)
        if block not in VALID_CUDA_BLOCKS:
            raise ValidationError(f"Unsupported DVEB CUDA block override {block}")
        if block == 0 and _CUDA_BLOCK_BY_STATE[phi.shape[1]] == 0:
            raise ValidationError(
                f"No calibrated automatic CUDA block exists for state {phi.shape[1]}"
            )
        nll, sigma2, status = _outputs(z.shape[0])
        selected = ctypes.c_int(-1)
        code = self.library.cdll.dveb_recurrence_cuda_run_host(
            self.context.pointer,
            f64_pointer(z), z.shape[0], z.shape[1],
            f64_pointer(phi), phi.shape[0], phi.shape[1],
            f64_pointer(loading), loading.shape[0], loading.shape[1],
            f64_pointer(nll), nll.size, f64_pointer(sigma2), sigma2.size,
            i64_pointer(status), status.size, block,
            ctypes.byref(selected),
        )
        self.library.raise_status(code, "CUDA transfer evaluation")
        self.last_selected_block = int(selected.value)
        return nll, sigma2, status.astype(np.bool_)

    def close(self) -> None:
        self.context.close()

    def __enter__(self):
        self.context.require_open()
        return self

    def __exit__(self, *_args):
        self.close()
