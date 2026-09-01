"""Verified ctypes binding for the DVEB dense CPU ABI v1.

The research branch carries one immutable, Forge-qualified library.  Loading
is deliberately strict: platform, file hash, ABI version, and every status are
checked.  There is no system-library search and no fallback implementation.
"""

from __future__ import annotations

import ctypes
import hashlib
import platform
import sys
from contextlib import suppress
from pathlib import Path

import numpy as np

from pystatistics.core.exceptions import NumericalError, ValidationError

ABI_VERSION = 1
LIBRARY_SHA256 = "a96e410282fae692f532185ba9c6d0377885f9f7f2b696ff772eda13b1fb102f"
LIBRARY_PATH = Path(__file__).with_name("artifacts") / "mvnmle_cpu_abi_v1.so"

SCHEDULE_AUTO = 0
SCHEDULE_SERIAL = 1
SCHEDULE_WORK_ITEM_PARALLEL = 2
VALID_SCHEDULES = {
    SCHEDULE_AUTO,
    SCHEDULE_SERIAL,
    SCHEDULE_WORK_ITEM_PARALLEL,
}

STATUS_OK = 0
STATUS_INVALID = 2
STATUS_NONFINITE = 3
STATUS_SHAPE = 4
STATUS_ALIAS = 5
STATUS_FACTORIZATION = 6
STATUS_ALLOCATION = 7
STATUS_SCHEDULE = 8

_F64_PTR = ctypes.POINTER(ctypes.c_double)
_I64_PTR = ctypes.POINTER(ctypes.c_int64)
_CTX_PTR = ctypes.c_void_p


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_platform() -> None:
    machine = platform.machine().lower()
    if not sys.platform.startswith("linux") or machine not in {"x86_64", "amd64"}:
        raise RuntimeError(
            "The bundled DVEB MVN-MLE research artifact is qualified only for "
            f"Linux x86-64 on Forge; got platform={sys.platform!r}, "
            f"machine={platform.machine()!r}. No fallback was attempted."
        )


def _f64_ptr(array: np.ndarray) -> _F64_PTR:
    return array.ctypes.data_as(_F64_PTR)


def _i64_ptr(array: np.ndarray) -> _I64_PTR:
    return array.ctypes.data_as(_I64_PTR)


class DenseLibrary:
    """One verified library handle and its typed ABI functions."""

    def __init__(self, path: str | Path | None = None):
        _require_platform()
        self.path = Path(path) if path is not None else LIBRARY_PATH
        if not self.path.is_file():
            raise RuntimeError(
                f"The DVEB MVN-MLE artifact is missing at {self.path}. "
                "The experimental solver cannot run; no fallback was attempted."
            )
        actual = _sha256(self.path)
        if actual != LIBRARY_SHA256:
            raise RuntimeError(
                "The DVEB MVN-MLE artifact failed its integrity check: "
                f"expected SHA-256 {LIBRARY_SHA256}, got {actual}."
            )

        try:
            self.cdll = ctypes.CDLL(str(self.path))
        except OSError as exc:
            raise RuntimeError(
                f"The qualified DVEB artifact could not be loaded from {self.path}: {exc}"
            ) from exc

        self.cdll.dveb_dense_abi_version.argtypes = []
        self.cdll.dveb_dense_abi_version.restype = ctypes.c_uint32
        self.cdll.dveb_dense_status_string.argtypes = [ctypes.c_int]
        self.cdll.dveb_dense_status_string.restype = ctypes.c_char_p
        self.cdll.dveb_dense_context_create.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            _I64_PTR,
            ctypes.c_size_t,
            _I64_PTR,
            ctypes.c_size_t,
            _I64_PTR,
            ctypes.c_size_t,
            _F64_PTR,
            ctypes.c_size_t,
            _F64_PTR,
            ctypes.c_size_t,
            _F64_PTR,
            ctypes.c_size_t,
            _F64_PTR,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_size_t,
            ctypes.POINTER(_CTX_PTR),
        ]
        self.cdll.dveb_dense_context_create.restype = ctypes.c_int
        self.cdll.dveb_dense_value_gradient.argtypes = [
            _CTX_PTR,
            _F64_PTR,
            ctypes.c_size_t,
            _F64_PTR,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.cdll.dveb_dense_value_gradient.restype = ctypes.c_int
        self.cdll.dveb_dense_context_destroy.argtypes = [_CTX_PTR]
        self.cdll.dveb_dense_context_destroy.restype = None
        self.cdll.dveb_dense_context_scratch_bytes.argtypes = [_CTX_PTR]
        self.cdll.dveb_dense_context_scratch_bytes.restype = ctypes.c_size_t

        version = int(self.cdll.dveb_dense_abi_version())
        if version != ABI_VERSION:
            raise RuntimeError(
                f"DVEB dense ABI mismatch: expected {ABI_VERSION}, got {version}."
            )

    def status_text(self, status: int) -> str:
        raw = self.cdll.dveb_dense_status_string(int(status))
        return raw.decode("utf-8", errors="replace") if raw else f"status {status}"

    def raise_status(self, status: int, operation: str) -> None:
        if status == STATUS_OK:
            return
        message = f"DVEB {operation} failed: {self.status_text(status)} (status {status})"
        if status in {STATUS_INVALID, STATUS_SHAPE, STATUS_ALIAS, STATUS_SCHEDULE}:
            raise ValidationError(message)
        if status in {STATUS_NONFINITE, STATUS_FACTORIZATION, STATUS_ALLOCATION}:
            raise NumericalError(message)
        raise RuntimeError(message)


class DenseContext:
    """Owned native context; caller arrays are never retained by the ABI."""

    def __init__(
        self,
        library: DenseLibrary,
        *,
        p: int,
        patterns: int,
        offsets: np.ndarray,
        centered_offsets: np.ndarray,
        observed_index: np.ndarray,
        n_k: np.ndarray,
        ybar: np.ndarray,
        centered: np.ndarray,
        jitter_scale: np.ndarray,
        epsilon: float,
        max_threads: int,
    ):
        self.library = library
        self._pointer = _CTX_PTR()
        status = library.cdll.dveb_dense_context_create(
            p,
            patterns,
            _i64_ptr(offsets),
            offsets.size,
            _i64_ptr(centered_offsets),
            centered_offsets.size,
            _i64_ptr(observed_index),
            observed_index.size,
            _f64_ptr(n_k),
            n_k.size,
            _f64_ptr(ybar),
            ybar.size,
            _f64_ptr(centered),
            centered.size,
            _f64_ptr(jitter_scale),
            jitter_scale.size,
            float(epsilon),
            max_threads,
            ctypes.byref(self._pointer),
        )
        library.raise_status(status, "context creation")
        if not self._pointer:
            raise RuntimeError("DVEB context creation returned success with a null context")

    @property
    def closed(self) -> bool:
        return not bool(self._pointer)

    @property
    def scratch_bytes(self) -> int:
        self._require_open()
        return int(self.library.cdll.dveb_dense_context_scratch_bytes(self._pointer))

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("The DVEB MVN-MLE objective context is closed")

    def value_gradient(
        self,
        theta: np.ndarray,
        gradient: np.ndarray,
        *,
        threads: int,
        schedule: int,
    ) -> tuple[float, int]:
        self._require_open()
        value = ctypes.c_double(np.nan)
        selected = ctypes.c_int(-1)
        status = self.library.cdll.dveb_dense_value_gradient(
            self._pointer,
            _f64_ptr(theta),
            theta.size,
            _f64_ptr(gradient),
            gradient.size,
            threads,
            schedule,
            ctypes.byref(value),
            ctypes.byref(selected),
        )
        self.library.raise_status(status, "value-and-gradient evaluation")
        return float(value.value), int(selected.value)

    def close(self) -> None:
        if self._pointer:
            self.library.cdll.dveb_dense_context_destroy(self._pointer)
            self._pointer = _CTX_PTR()

    def __enter__(self) -> DenseContext:
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
