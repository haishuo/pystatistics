"""ctypes boundary for the separately authored exact-ARMA diagnostics."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)
BYTE_POINTER = ctypes.POINTER(ctypes.c_uint8)


def _double_pointer(array: NDArray) -> DOUBLE_POINTER:
    return array.ctypes.data_as(DOUBLE_POINTER)


def _byte_pointer(array: NDArray) -> BYTE_POINTER:
    return array.ctypes.data_as(BYTE_POINTER)


def _input(array: NDArray, *, ndim: int) -> NDArray[np.float64]:
    result = np.asarray(array)
    if result.dtype != np.float64 or result.ndim != ndim or not result.flags.c_contiguous:
        raise ValueError(f"expected C-contiguous float64 rank-{ndim} input")
    return result


class NativeCPU:
    """Reusable NC context."""

    def __init__(self, library: Path, *, max_r: int = 25, max_threads: int = 12):
        self.library = ctypes.CDLL(str(library))
        self.library.arma_cpu_create.argtypes = [ctypes.c_int, ctypes.c_int]
        self.library.arma_cpu_create.restype = ctypes.c_void_p
        self.library.arma_cpu_destroy.argtypes = [ctypes.c_void_p]
        self.library.arma_cpu_evaluate.argtypes = [
            ctypes.c_void_p,
            DOUBLE_POINTER,
            DOUBLE_POINTER,
            DOUBLE_POINTER,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
            DOUBLE_POINTER,
            DOUBLE_POINTER,
            BYTE_POINTER,
        ]
        self.library.arma_cpu_evaluate.restype = ctypes.c_int
        self.context = self.library.arma_cpu_create(max_r, max_threads)
        if not self.context:
            raise RuntimeError("arma_cpu_create refused")

    def close(self) -> None:
        if self.context:
            self.library.arma_cpu_destroy(self.context)
            self.context = None

    def __enter__(self) -> NativeCPU:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def evaluate(
        self, z: NDArray, phi: NDArray, loading: NDArray, *, threads: int
    ) -> tuple[NDArray, NDArray, NDArray]:
        z = _input(z, ndim=2)
        phi = _input(phi, ndim=2)
        loading = _input(loading, ndim=2)
        k, n = z.shape
        if phi.shape != loading.shape or phi.shape[0] != k:
            raise ValueError("z/phi/loading batch shapes disagree")
        r = phi.shape[1]
        nll = np.empty(k, dtype=np.float64)
        sigma2 = np.empty(k, dtype=np.float64)
        status = np.empty(k, dtype=np.uint8)
        code = self.library.arma_cpu_evaluate(
            self.context,
            _double_pointer(z),
            _double_pointer(phi),
            _double_pointer(loading),
            k,
            n,
            r,
            threads,
            _double_pointer(nll),
            _double_pointer(sigma2),
            _byte_pointer(status),
        )
        if code != 0:
            raise RuntimeError(f"arma_cpu_evaluate refused with code {code}")
        return nll, sigma2, status.astype(np.bool_)


class NativeCUDA:
    """Shape-fixed NG context with explicit residency operations."""

    def __init__(
        self,
        library: Path,
        z: NDArray,
        phi: NDArray,
        loading: NDArray,
        *,
        phi_trace: NDArray | None = None,
        loading_trace: NDArray | None = None,
    ):
        self.library = ctypes.CDLL(str(library))
        self._bind()
        self.z = _input(z, ndim=2)
        self.phi = _input(phi, ndim=2)
        self.loading = _input(loading, ndim=2)
        self.k, self.n = self.z.shape
        if self.phi.shape != self.loading.shape or self.phi.shape[0] != self.k:
            raise ValueError("z/phi/loading batch shapes disagree")
        self.r = self.phi.shape[1]
        if (phi_trace is None) != (loading_trace is None):
            raise ValueError("both proposal traces are required together")
        self.phi_trace = None if phi_trace is None else _input(phi_trace, ndim=3)
        self.loading_trace = None if loading_trace is None else _input(loading_trace, ndim=3)
        if self.phi_trace is not None:
            if self.phi_trace.shape != self.loading_trace.shape:
                raise ValueError("proposal trace shapes disagree")
            if self.phi_trace.shape[1:] != self.phi.shape:
                raise ValueError("proposal trace/base shapes disagree")
            self.proposals = self.phi_trace.shape[0]
        else:
            self.proposals = 0
        self.context = self.library.arma_cuda_create(self.k, self.n, self.r, self.proposals)
        if not self.context:
            raise RuntimeError(f"arma_cuda_create refused: {self.last_error()}")
        self.upload_base()
        if self.phi_trace is not None:
            self.upload_trace()

    def _bind(self) -> None:
        library = self.library
        library.arma_cuda_create.argtypes = [
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.arma_cuda_create.restype = ctypes.c_void_p
        library.arma_cuda_destroy.argtypes = [ctypes.c_void_p]
        library.arma_cuda_upload_base.argtypes = [
            ctypes.c_void_p,
            DOUBLE_POINTER,
            DOUBLE_POINTER,
            DOUBLE_POINTER,
        ]
        library.arma_cuda_upload_trace.argtypes = [
            ctypes.c_void_p,
            DOUBLE_POINTER,
            DOUBLE_POINTER,
        ]
        library.arma_cuda_launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.arma_cuda_synchronize.argtypes = [ctypes.c_void_p]
        library.arma_cuda_download.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            DOUBLE_POINTER,
            DOUBLE_POINTER,
            BYTE_POINTER,
        ]
        library.arma_cuda_last_error.restype = ctypes.c_char_p

    def last_error(self) -> str:
        return self.library.arma_cuda_last_error().decode()

    def _check(self, code: int, operation: str) -> None:
        if code != 0:
            raise RuntimeError(f"{operation} refused with code {code}: {self.last_error()}")

    def upload_base(self) -> None:
        self._check(
            self.library.arma_cuda_upload_base(
                self.context,
                _double_pointer(self.z),
                _double_pointer(self.phi),
                _double_pointer(self.loading),
            ),
            "arma_cuda_upload_base",
        )

    def upload_trace(self) -> None:
        if self.phi_trace is None or self.loading_trace is None:
            raise ValueError("no proposal trace configured")
        self._check(
            self.library.arma_cuda_upload_trace(
                self.context,
                _double_pointer(self.phi_trace),
                _double_pointer(self.loading_trace),
            ),
            "arma_cuda_upload_trace",
        )

    def launch(self, *, proposal: int = -1, block_size: int) -> None:
        self._check(
            self.library.arma_cuda_launch(self.context, proposal, block_size),
            "arma_cuda_launch",
        )

    def synchronize(self) -> None:
        self._check(
            self.library.arma_cuda_synchronize(self.context),
            "arma_cuda_synchronize",
        )

    def download(self, *, proposal: int = -1) -> tuple[NDArray, NDArray, NDArray]:
        nll = np.empty(self.k, dtype=np.float64)
        sigma2 = np.empty(self.k, dtype=np.float64)
        status = np.empty(self.k, dtype=np.uint8)
        self._check(
            self.library.arma_cuda_download(
                self.context,
                proposal,
                _double_pointer(nll),
                _double_pointer(sigma2),
                _byte_pointer(status),
            ),
            "arma_cuda_download",
        )
        return nll, sigma2, status.astype(np.bool_)

    def evaluate(self, *, proposal: int = -1, block_size: int) -> tuple[NDArray, NDArray, NDArray]:
        self.launch(proposal=proposal, block_size=block_size)
        self.synchronize()
        return self.download(proposal=proposal)

    def close(self) -> None:
        if self.context:
            self.library.arma_cuda_destroy(self.context)
            self.context = None

    def __enter__(self) -> NativeCUDA:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
