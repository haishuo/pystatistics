"""Strict ctypes bindings for the frozen DVEB recurrence ABI v1."""

from __future__ import annotations

import ctypes
import hashlib
import json
import platform
import sys
from contextlib import suppress
from pathlib import Path

from pystatistics.core.exceptions import NumericalError, ValidationError

ABI_VERSION = 1
ARTIFACTS_PATH = Path(__file__).with_name("artifacts")
MANIFEST_PATH = ARTIFACTS_PATH / "manifest.json"


def _read_manifest() -> dict:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"The bundled DVEB exact-ARMA manifest is unreadable at {MANIFEST_PATH}: {exc}"
        ) from exc
    required = {
        "abi_version", "cpu_artifact", "cpu_artifact_sha256",
        "combined_artifact_sha256", "cpu_isa",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise RuntimeError(f"The bundled DVEB exact-ARMA manifest is missing {missing}")
    if manifest["abi_version"] != ABI_VERSION:
        raise RuntimeError(
            f"DVEB recurrence manifest ABI {manifest['abi_version']!r} != {ABI_VERSION}"
        )
    if manifest["cpu_isa"] != "x86-64-v2":
        raise RuntimeError(
            f"Unsupported DVEB CPU ISA {manifest['cpu_isa']!r}; expected 'x86-64-v2'"
        )
    return manifest


MANIFEST = _read_manifest()
CPU_LIBRARY_PATH = ARTIFACTS_PATH / str(MANIFEST["cpu_artifact"])
CPU_LIBRARY_SHA256 = str(MANIFEST["cpu_artifact_sha256"])
COMBINED_LIBRARY_SHA256 = str(MANIFEST["combined_artifact_sha256"])

STATUS_OK = 0
STATUS_INVALID = 2
STATUS_SHAPE = 3
STATUS_ALIAS = 4
STATUS_ALLOCATION = 5
STATUS_SCHEDULE = 6
STATUS_CUDA = 7

CPU_AUTO = 0
CPU_SERIAL = 1
CPU_ITEM_PARALLEL = 2
VALID_CPU_SCHEDULES = {CPU_SERIAL, CPU_ITEM_PARALLEL}
VALID_CUDA_BLOCKS = {0, 32, 64, 128, 256}

_F64 = ctypes.POINTER(ctypes.c_double)
_I64 = ctypes.POINTER(ctypes.c_int64)
_INT = ctypes.POINTER(ctypes.c_int)
_CTX = ctypes.c_void_p
_SIZE = ctypes.c_size_t


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_X86_64_V2_FEATURES = frozenset(
    {"cx16", "lahf_lm", "popcnt", "sse3", "ssse3", "sse4_1", "sse4_2"}
)


def _linux_cpu_flags(path: Path = Path("/proc/cpuinfo")) -> frozenset[str] | None:
    try:
        for line in path.read_text().splitlines():
            name, separator, values = line.partition(":")
            if separator and name.strip().lower() in {"flags", "features"}:
                flags = {value.lower() for value in values.split()}
                if "pni" in flags:
                    flags.add("sse3")
                return frozenset(flags)
    except OSError:
        return None
    return None


def _missing_x86_64_v2_features(flags: frozenset[str] | None) -> frozenset[str]:
    if flags is None:
        return _X86_64_V2_FEATURES
    normalized = set(flags)
    if "pni" in normalized:
        normalized.add("sse3")
    return _X86_64_V2_FEATURES.difference(normalized)


def _require_linux_x86_64_v2() -> None:
    if not sys.platform.startswith("linux") or platform.machine().lower() not in {
        "x86_64", "amd64",
    }:
        raise RuntimeError(
            "The DVEB exact-ARMA CPU artifact is qualified only for Linux x86-64; "
            f"got {sys.platform!r}/{platform.machine()!r}. No fallback was attempted."
        )
    missing = sorted(_missing_x86_64_v2_features(_linux_cpu_flags()))
    if missing:
        raise RuntimeError(
            f"The DVEB artifact requires x86-64-v2; missing {missing}. "
            "No fallback was attempted."
        )


class RecurrenceLibrary:
    """A verified CPU-only or combined recurrence library handle."""

    def __init__(self, *, cuda: bool, path: str | Path | None = None):
        self.cuda = bool(cuda)
        if self.cuda:
            if path is None:
                raise RuntimeError(
                    "The CUDA-linked DVEB research artifact is not bundled with the CPU "
                    "package; an explicit qualified path is required. No fallback was attempted."
                )
            self.path = Path(path)
            expected = COMBINED_LIBRARY_SHA256
        else:
            _require_linux_x86_64_v2()
            self.path = CPU_LIBRARY_PATH if path is None else Path(path)
            expected = CPU_LIBRARY_SHA256
        if not self.path.is_file():
            raise RuntimeError(
                f"The DVEB exact-ARMA artifact is missing at {self.path}. "
                "No fallback was attempted."
            )
        actual = _sha256(self.path)
        if actual != expected:
            raise RuntimeError(
                f"The DVEB exact-ARMA artifact failed integrity checking: "
                f"expected {expected}, got {actual}."
            )
        try:
            self.cdll = ctypes.CDLL(str(self.path))
        except OSError as exc:
            raise RuntimeError(
                f"The qualified DVEB exact-ARMA artifact could not be loaded: {exc}"
            ) from exc
        self._bind_common()
        if self.cuda:
            self._bind_cuda()
        version = int(self.cdll.dveb_recurrence_abi_version())
        if version != ABI_VERSION:
            raise RuntimeError(f"DVEB recurrence ABI mismatch: expected 1, got {version}")

    def _bind_common(self) -> None:
        lib = self.cdll
        lib.dveb_recurrence_abi_version.argtypes = []
        lib.dveb_recurrence_abi_version.restype = ctypes.c_uint32
        lib.dveb_recurrence_status_string.argtypes = [ctypes.c_int]
        lib.dveb_recurrence_status_string.restype = ctypes.c_char_p
        lib.dveb_recurrence_cpu_context_create.argtypes = [_SIZE, ctypes.POINTER(_CTX)]
        lib.dveb_recurrence_cpu_context_create.restype = ctypes.c_int
        lib.dveb_recurrence_cpu_context_destroy.argtypes = [_CTX]
        lib.dveb_recurrence_cpu_context_destroy.restype = None
        lib.dveb_recurrence_cpu_scratch_bytes.argtypes = [_CTX]
        lib.dveb_recurrence_cpu_scratch_bytes.restype = _SIZE
        lib.dveb_recurrence_cpu_run.argtypes = [
            _CTX, _F64, _SIZE, _SIZE, _F64, _SIZE, _SIZE,
            _F64, _SIZE, _SIZE, _F64, _SIZE, _F64, _SIZE,
            _I64, _SIZE, _SIZE, ctypes.c_int, _INT,
        ]
        lib.dveb_recurrence_cpu_run.restype = ctypes.c_int

    def _bind_cuda(self) -> None:
        lib = self.cdll
        lib.dveb_recurrence_cuda_context_create.argtypes = [
            ctypes.c_int, _SIZE, _SIZE, _INT, _SIZE, ctypes.POINTER(_CTX),
        ]
        lib.dveb_recurrence_cuda_context_create.restype = ctypes.c_int
        lib.dveb_recurrence_cuda_context_destroy.argtypes = [_CTX]
        lib.dveb_recurrence_cuda_context_destroy.restype = None
        lib.dveb_recurrence_cuda_payload_bytes.argtypes = [_CTX]
        lib.dveb_recurrence_cuda_payload_bytes.restype = _SIZE
        lib.dveb_recurrence_cuda_run_host.argtypes = [
            _CTX, _F64, _SIZE, _SIZE, _F64, _SIZE, _SIZE,
            _F64, _SIZE, _SIZE, _F64, _SIZE, _F64, _SIZE,
            _I64, _SIZE, ctypes.c_int, _INT,
        ]
        lib.dveb_recurrence_cuda_run_host.restype = ctypes.c_int

    def status_text(self, status: int) -> str:
        raw = self.cdll.dveb_recurrence_status_string(int(status))
        return raw.decode(errors="replace") if raw else f"status {status}"

    def raise_status(self, status: int, operation: str) -> None:
        if status == STATUS_OK:
            return
        message = f"DVEB {operation} failed: {self.status_text(status)} (status {status})"
        if status in {STATUS_INVALID, STATUS_SHAPE, STATUS_ALIAS, STATUS_SCHEDULE}:
            raise ValidationError(message)
        if status == STATUS_ALLOCATION:
            raise NumericalError(message)
        if status == STATUS_CUDA:
            raise RuntimeError(message + "; no CPU fallback was attempted")
        raise RuntimeError(message)


class CPUContext:
    def __init__(self, library: RecurrenceLibrary, max_threads: int):
        self.library = library
        self.pointer = _CTX()
        status = library.cdll.dveb_recurrence_cpu_context_create(
            max_threads, ctypes.byref(self.pointer)
        )
        library.raise_status(status, "CPU context creation")
        if not self.pointer:
            raise RuntimeError("DVEB CPU context creation returned a null context")

    @property
    def scratch_bytes(self) -> int:
        self.require_open()
        return int(self.library.cdll.dveb_recurrence_cpu_scratch_bytes(self.pointer))

    def require_open(self) -> None:
        if not self.pointer:
            raise RuntimeError("The DVEB exact-ARMA CPU context is closed")

    def close(self) -> None:
        if self.pointer:
            self.library.cdll.dveb_recurrence_cpu_context_destroy(self.pointer)
            self.pointer = _CTX()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


class CUDAHostContext:
    def __init__(
        self, library: RecurrenceLibrary, *, max_f64: int, max_i64: int,
        block_by_state: tuple[int, ...], device: int,
    ):
        self.library = library
        self.pointer = _CTX()
        mapping = (ctypes.c_int * len(block_by_state))(*block_by_state)
        status = library.cdll.dveb_recurrence_cuda_context_create(
            device, max_f64, max_i64, mapping, len(mapping), ctypes.byref(self.pointer)
        )
        library.raise_status(status, "CUDA transfer context creation")
        if not self.pointer:
            raise RuntimeError("DVEB CUDA context creation returned a null context")

    @property
    def payload_bytes(self) -> int:
        self.require_open()
        return int(self.library.cdll.dveb_recurrence_cuda_payload_bytes(self.pointer))

    def require_open(self) -> None:
        if not self.pointer:
            raise RuntimeError("The DVEB exact-ARMA CUDA context is closed")

    def close(self) -> None:
        if self.pointer:
            self.library.cdll.dveb_recurrence_cuda_context_destroy(self.pointer)
            self.pointer = _CTX()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def f64_pointer(array):
    return array.ctypes.data_as(_F64)


def i64_pointer(array):
    return array.ctypes.data_as(_I64)
