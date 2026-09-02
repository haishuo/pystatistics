"""Deterministic batching for independent SciPy optimizer states."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from .common import MAX_ACTIVE_OPTIMIZERS, THREAD_STACK_BYTES

OPTIMIZER_OPTIONS = {
    "ftol": 1.0e-8,
    "gtol": 1.0e-5,
    "maxiter": 300,
    "maxfun": 15_000,
    "maxcor": 10,
    "maxls": 20,
}


class GradientBatchBackend(Protocol):
    def value_and_gradient(
        self, rows: tuple[int, ...], parameters: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


@dataclass
class _Slot:
    state: str = "running"
    parameters: NDArray[np.float64] | None = None
    value: float | None = None
    gradient: NDArray[np.float64] | None = None
    error: BaseException | None = None


@dataclass(frozen=True)
class CoordinatorAccounting:
    barrier_batches: int
    likelihood_rows: int
    request_rows: int
    maximum_batch_rows: int


class DeterministicCoordinator:
    """Batch `(f,g)` calls without changing any optimizer's call sequence."""

    def __init__(self, backend: GradientBatchBackend):
        self.backend = backend

    def fit_chunk(
        self, starts: NDArray[np.float64], *, row_offset: int = 0
    ) -> tuple[list[OptimizeResult], CoordinatorAccounting]:
        starts = np.ascontiguousarray(starts, dtype=np.float64)
        count = starts.shape[0]
        if not 1 <= count <= MAX_ACTIVE_OPTIMIZERS:
            raise ValueError(f"coordinator chunk size must be in [1,256], got {count}")
        condition = threading.Condition()
        slots = [_Slot() for _ in range(count)]
        results: list[OptimizeResult | None] = [None] * count

        previous_stack = threading.stack_size()
        threading.stack_size(THREAD_STACK_BYTES)

        def request(local_row: int, values: NDArray[np.float64]):
            with condition:
                slot = slots[local_row]
                if slot.state != "running":
                    raise RuntimeError(f"worker {local_row} submitted from state {slot.state}")
                slot.parameters = np.array(values, dtype=np.float64, copy=True, order="C")
                slot.state = "request"
                condition.notify_all()
                condition.wait_for(lambda: slot.state in {"response", "error"})
                if slot.state == "error":
                    assert slot.error is not None
                    raise slot.error
                assert slot.value is not None and slot.gradient is not None
                value = slot.value
                gradient = slot.gradient.copy()
                slot.state = "running"
                slot.parameters = None
                slot.value = None
                slot.gradient = None
                condition.notify_all()
                return value, gradient

        def worker(local_row: int):
            try:
                result = minimize(
                    lambda values: request(local_row, values),
                    starts[local_row],
                    method="L-BFGS-B",
                    jac=True,
                    options=OPTIMIZER_OPTIONS,
                )
                results[local_row] = result
            except BaseException as exc:  # propagate the original backend failure
                results[local_row] = OptimizeResult(
                    x=starts[local_row].copy(), fun=np.inf, success=False,
                    status=-999, message=f"coordinator worker exception: {exc!r}",
                    nit=0, nfev=0, njev=0, exception=exc,
                )
            finally:
                with condition:
                    slots[local_row].state = "done"
                    condition.notify_all()

        threads = [
            threading.Thread(target=worker, args=(row,), name=f"dveb-fit-{row_offset + row}")
            for row in range(count)
        ]
        for thread in threads:
            thread.start()

        barrier_batches = 0
        likelihood_rows = 0
        request_rows = 0
        maximum_batch_rows = 0
        live = set(range(count))
        try:
            while live:
                with condition:
                    condition.wait_for(
                        lambda: all(slots[row].state in {"request", "done"} for row in live)
                    )
                    done = {row for row in live if slots[row].state == "done"}
                    live.difference_update(done)
                    requested = tuple(row for row in sorted(live) if slots[row].state == "request")
                    if not requested:
                        continue
                    parameters = np.stack(
                        [slots[row].parameters for row in requested], axis=0
                    )
                try:
                    values, gradients = self.backend.value_and_gradient(
                        tuple(row_offset + row for row in requested), parameters
                    )
                    values = np.asarray(values, dtype=np.float64)
                    gradients = np.asarray(gradients, dtype=np.float64)
                    if values.shape != (len(requested),) or gradients.shape != parameters.shape:
                        raise RuntimeError(
                            f"backend returned shapes {values.shape}/{gradients.shape}, "
                            f"expected {(len(requested),)}/{parameters.shape}"
                        )
                except BaseException as exc:
                    with condition:
                        for row in requested:
                            slots[row].error = exc
                            slots[row].state = "error"
                        condition.notify_all()
                    continue
                with condition:
                    for output_row, local_row in enumerate(requested):
                        slots[local_row].value = float(values[output_row])
                        slots[local_row].gradient = np.array(
                            gradients[output_row], dtype=np.float64, copy=True
                        )
                        slots[local_row].state = "response"
                    condition.notify_all()
                rows_used = int(getattr(self.backend, "last_likelihood_rows", len(requested)))
                barrier_batches += 1
                request_rows += len(requested)
                likelihood_rows += rows_used
                maximum_batch_rows = max(maximum_batch_rows, rows_used)
        finally:
            for thread in threads:
                thread.join()
            threading.stack_size(previous_stack)

        if any(result is None for result in results):
            raise RuntimeError("coordinator lost an optimizer result")
        return [result for result in results if result is not None], CoordinatorAccounting(
            barrier_batches=barrier_batches,
            likelihood_rows=likelihood_rows,
            request_rows=request_rows,
            maximum_batch_rows=maximum_batch_rows,
        )

    def fit(
        self, starts: NDArray[np.float64]
    ) -> tuple[list[OptimizeResult], CoordinatorAccounting]:
        starts = np.ascontiguousarray(starts, dtype=np.float64)
        results = []
        accounting = []
        for offset in range(0, starts.shape[0], MAX_ACTIVE_OPTIMIZERS):
            chunk_results, chunk_accounting = self.fit_chunk(
                starts[offset : offset + MAX_ACTIVE_OPTIMIZERS], row_offset=offset
            )
            results.extend(chunk_results)
            accounting.append(chunk_accounting)
        return results, CoordinatorAccounting(
            barrier_batches=sum(item.barrier_batches for item in accounting),
            likelihood_rows=sum(item.likelihood_rows for item in accounting),
            request_rows=sum(item.request_rows for item in accounting),
            maximum_batch_rows=max(item.maximum_batch_rows for item in accounting),
        )
