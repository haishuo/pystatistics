"""Persistent existing-tool contexts for the exact-ARMA campaign."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
from reference_impl import KAPPA, _finish

from pystatistics.timeseries._arima_kalman_kernel import ArmaKalmanWorkspace


class CythonParallel:
    """C2: static series partitions over the existing GIL-free Cython core."""

    def __init__(self, z: NDArray, *, max_threads: int = 12):
        self.z = np.asarray(z)
        if self.z.dtype != np.float64 or self.z.ndim != 2 or not self.z.flags.c_contiguous:
            raise ValueError("C2 requires C-contiguous float64 z[K,n]")
        self.k, self.n = self.z.shape
        self.max_threads = max_threads
        self.workspaces: dict[int, list[ArmaKalmanWorkspace]] = {}
        self.executors: dict[int, ThreadPoolExecutor] = {}

    def _workspace(self, r: int) -> list[ArmaKalmanWorkspace]:
        if r not in self.workspaces:
            self.workspaces[r] = [ArmaKalmanWorkspace(r, self.n) for _ in range(self.k)]
        return self.workspaces[r]

    def evaluate(
        self, phi: NDArray, loading: NDArray, *, threads: int
    ) -> tuple[NDArray, NDArray, NDArray]:
        if threads not in (1, 6, 12) or threads > self.max_threads:
            raise ValueError(f"unsupported C2 thread count: {threads}")
        if phi.shape != loading.shape or phi.shape[0] != self.k:
            raise ValueError("C2 parameter shape mismatch")
        r = phi.shape[1]
        workspaces = self._workspace(r)
        nll = np.empty(self.k, dtype=np.float64)
        sigma2 = np.empty(self.k, dtype=np.float64)
        status = np.empty(self.k, dtype=np.bool_)

        def run(start: int, stop: int) -> None:
            for row in range(start, stop):
                ok, sse, sum_log_f = workspaces[row].loglik_parts(
                    self.z[row], phi[row], loading[row], KAPPA
                )
                nll[row], sigma2[row], status[row] = _finish(self.n, ok, sse, sum_log_f)

        if threads == 1:
            run(0, self.k)
        else:
            executor = self.executors.setdefault(threads, ThreadPoolExecutor(max_workers=threads))
            bounds = [
                (self.k * index // threads, self.k * (index + 1) // threads)
                for index in range(threads)
            ]
            futures = [executor.submit(run, start, stop) for start, stop in bounds if start < stop]
            for future in futures:
                future.result()
        return nll, sigma2, status

    def close(self) -> None:
        for executor in self.executors.values():
            executor.shutdown()
        self.executors.clear()

    def __enter__(self) -> CythonParallel:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
