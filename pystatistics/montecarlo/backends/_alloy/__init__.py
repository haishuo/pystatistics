"""Loading the vendored ALLOY runtime and bootstrap bundle.

WHAT IS VENDORED AND WHY. ``artifacts/`` holds a prebuilt ALLOY runtime and one
compiled bundle, checked in and shipped as package data. PyStatistics therefore
needs no ALLOY checkout, no compiler, and no build step at install time -- which
is the whole point: a user who ``pip install``s this package gets the Metal
bootstrap or gets a clear refusal, never a dependency on a developer's machine.

THE ARTIFACTS ARE LOADED FROM HERE AND NOWHERE ELSE. There is deliberately no
environment variable, no search of a source tree, no walk up the filesystem and
no fallback to a system location. Every one of those turns "which code ran?"
into a question about the machine rather than about the package, and a numerical
result whose provenance depends on the machine is not reproducible. If the
packaged files are absent or do not match their recorded digests, this module
raises; it does not look elsewhere.

``artifacts.json`` records the ALLOY commit, the compiler and bundle-format
versions, the source digest and every binary digest. ``PROVENANCE.md`` says how
to rebuild them.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import platform
import sys
import threading
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ARTIFACTS = _HERE / "artifacts"
_RECORD = _HERE / "artifacts.json"

ALLOY_OK = 0
TARGET_CPU, TARGET_GPU = 0, 1

#: Bundle-format major.minor this loader understands. The runtime does its own
#: full validation at open; this is the earlier, clearer refusal for a bundle
#: that was rebuilt against a format this release was never tested with.
SUPPORTED_BUNDLE_FORMAT = (0, 10)


class AlloyUnavailable(RuntimeError):
    """The packaged ALLOY backend cannot be used, with the reason stated."""


def _require_platform() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise AlloyUnavailable(
            "the packaged ALLOY bootstrap backend is built for macOS on Apple "
            f"Silicon (darwin/arm64); this is {sys.platform}/"
            f"{platform.machine()}. Use backend='cpu', or backend='gpu' on a "
            "CUDA machine, both of which are unaffected."
        )


def _read_record() -> dict:
    if not _RECORD.is_file():
        raise AlloyUnavailable(
            f"the ALLOY artifact record is missing ({_RECORD}). The package was "
            "installed without its packaged binaries; reinstall a wheel built "
            "for macOS arm64."
        )
    return json.loads(_RECORD.read_text())


def _verify(record: dict) -> None:
    """Every packaged file present and byte-for-byte what was recorded.

    Not defensive decoration. These binaries decide numerical results, and a
    truncated or half-updated install is otherwise discovered as a crash inside
    a dynamic loader or, worse, as a wrong answer.
    """
    for rel, want in sorted(record["artifacts"].items()):
        path = _ARTIFACTS / rel
        if not path.is_file():
            raise AlloyUnavailable(
                f"packaged ALLOY artifact is missing: {rel}. Reinstall a wheel "
                "built for macOS arm64."
            )
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise AlloyUnavailable(
                f"packaged ALLOY artifact {rel} does not match its recorded "
                f"digest (expected {want[:16]}..., found {got[:16]}...). The "
                "install is inconsistent; reinstall the package."
            )
    fmt = tuple(int(x) for x in record["bundle_format_version"].split(".")[:2])
    if fmt != SUPPORTED_BUNDLE_FORMAT:
        raise AlloyUnavailable(
            f"packaged bundle is format {record['bundle_format_version']}, and "
            f"this release supports {SUPPORTED_BUNDLE_FORMAT[0]}."
            f"{SUPPORTED_BUNDLE_FORMAT[1]}.x."
        )


# --------------------------------------------------------------- ctypes ABI

class _Error(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int), ("message", ctypes.c_char * 512)]


class _DeviceInfo(ctypes.Structure):
    _fields_ = [
        ("device_name", ctypes.c_char * 128),
        ("has_gpu", ctypes.c_bool),
        ("unified_memory", ctypes.c_bool),
        ("gpu_family", ctypes.c_char * 16),
        ("max_threads_per_threadgroup", ctypes.c_uint64),
    ]


class _View2Desc(ctypes.Structure):
    _fields_ = [
        ("elem_offset", ctypes.c_uint64),
        ("shape", ctypes.c_uint64 * 2),
        ("strides", ctypes.c_uint64 * 2),
    ]


class _Arg(ctypes.Structure):
    # The union's widest member is 8 bytes, so one u64 gives the right size and
    # alignment. It is never filled by hand -- the C constructors below produce
    # every value, which is what keeps the discriminant honest.
    _fields_ = [
        ("kind", ctypes.c_int),
        ("buffer", ctypes.c_void_p),
        ("elem_count", ctypes.c_uint64),
        ("scalar_type", ctypes.c_int),
        ("scalar", ctypes.c_uint64),
    ]


class _InvokeResult(ctypes.Structure):
    _fields_ = [("selected", ctypes.c_int), ("reason", ctypes.c_char * 256)]


_lib = None
_record = None


def _bind():
    global _lib, _record
    if _lib is not None:
        return _lib, _record
    _require_platform()
    rec = _read_record()
    _verify(rec)
    try:
        lib = ctypes.CDLL(str(_ARTIFACTS / "liballoy.dylib"))
    except OSError as exc:
        raise AlloyUnavailable(
            f"the packaged ALLOY runtime could not be loaded: {exc}"
        ) from exc

    lib.alloy_context_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(_Error)]
    lib.alloy_context_create.restype = ctypes.c_int
    lib.alloy_context_device_info.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_DeviceInfo), ctypes.POINTER(_Error)]
    lib.alloy_context_device_info.restype = ctypes.c_int
    lib.alloy_context_destroy.argtypes = [ctypes.c_void_p]
    lib.alloy_library_open.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_Error)]
    lib.alloy_library_open.restype = ctypes.c_int
    lib.alloy_library_close.argtypes = [ctypes.c_void_p]
    lib.alloy_buffer_create.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_Error)]
    lib.alloy_buffer_create.restype = ctypes.c_int
    lib.alloy_buffer_contents.argtypes = [ctypes.c_void_p]
    lib.alloy_buffer_contents.restype = ctypes.c_void_p
    lib.alloy_buffer_release.argtypes = [ctypes.c_void_p]
    lib.alloy_arg_span.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.alloy_arg_span.restype = _Arg
    lib.alloy_view2_contiguous.argtypes = [
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
    lib.alloy_view2_contiguous.restype = _View2Desc
    lib.alloy_arg_view2.argtypes = [ctypes.c_void_p, ctypes.POINTER(_View2Desc)]
    lib.alloy_arg_view2.restype = _Arg
    lib.alloy_arg_u64.argtypes = [ctypes.c_uint64]
    lib.alloy_arg_u64.restype = _Arg
    lib.alloy_arg_usize.argtypes = [ctypes.c_uint64]
    lib.alloy_arg_usize.restype = _Arg
    lib.alloy_invoke.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(_Arg),
        ctypes.c_size_t, ctypes.POINTER(_InvokeResult), ctypes.POINTER(_Error)]
    lib.alloy_invoke.restype = ctypes.c_int

    _lib, _record = lib, rec
    return _lib, _record


def provenance() -> dict:
    """What was vendored: ALLOY commit, versions, and artifact digests."""
    _bind()
    return dict(_record)


# --------------------------------------------------- the process-wide session
#
# THE CONTEXT AND THE BUNDLE ARE OPENED ONCE. Opening and closing them per call
# measured 5.7 ms on an M2 Max -- several times the cost of the bootstrap
# itself at the smaller pinned sizes -- and it buys nothing: the bundle is
# immutable package data and the context is a device handle. PyTorch keeps its
# device for the life of the process for the same reason.
#
# They are never closed. A device handle held to process exit is the normal
# shape for this; closing it would only reintroduce the cost at the next call.
_session_lock = threading.Lock()
_session = None


def shared_session():
    """(Context, Library) for the packaged bootstrap bundle, created once."""
    global _session
    with _session_lock:
        if _session is None:
            ctx = Context()
            _session = (ctx, Library(ctx, str(_ARTIFACTS / "bootstrap.alloylib")))
        return _session


#: Held across an invoke sequence. The runtime makes no documented thread-safety
#: promise, and there is one GPU, so serialising is both the safe reading and
#: very nearly free relative to the millisecond-scale work it guards.
invoke_lock = threading.Lock()


def is_available() -> tuple[bool, str]:
    """(usable, reason). Never raises, so a caller may ask before committing."""
    try:
        ctx = Context()
    except AlloyUnavailable as exc:
        return False, str(exc)
    try:
        if not ctx.has_gpu:
            return False, f"no usable Metal device ({ctx.device_name})"
        return True, ""
    finally:
        ctx.close()


class Context:
    def __init__(self) -> None:
        lib, _ = _bind()
        self._lib = lib
        self._ctx = ctypes.c_void_p()
        err = _Error()
        if lib.alloy_context_create(ctypes.byref(self._ctx),
                                    ctypes.byref(err)) != ALLOY_OK:
            raise AlloyUnavailable(
                f"ALLOY context: {err.message.decode(errors='replace')}")
        self.info = _DeviceInfo()
        if lib.alloy_context_device_info(self._ctx, ctypes.byref(self.info),
                                         ctypes.byref(err)) != ALLOY_OK:
            raise AlloyUnavailable(
                f"ALLOY device info: {err.message.decode(errors='replace')}")

    @property
    def has_gpu(self) -> bool:
        return bool(self.info.has_gpu)

    @property
    def device_name(self) -> str:
        return self.info.device_name.decode()

    def open_bootstrap(self) -> "Library":
        return Library(self, str(_ARTIFACTS / "bootstrap.alloylib"))

    def buffer(self, nbytes: int) -> "Buffer":
        return Buffer(self, nbytes)

    def close(self) -> None:
        if self._ctx:
            self._lib.alloy_context_destroy(self._ctx)
            self._ctx = ctypes.c_void_p()


class Buffer:
    def __init__(self, ctx: Context, nbytes: int) -> None:
        self._lib = ctx._lib
        self._h = ctypes.c_void_p()
        err = _Error()
        if self._lib.alloy_buffer_create(ctx._ctx, nbytes, ctypes.byref(self._h),
                                         ctypes.byref(err)) != ALLOY_OK:
            raise AlloyUnavailable(
                f"ALLOY buffer: {err.message.decode(errors='replace')}")
        self._n = nbytes

    def as_np(self, dtype, shape) -> np.ndarray:
        """A NumPy view of the buffer. Unified memory: no copy, no transfer."""
        ptr = self._lib.alloy_buffer_contents(self._h)
        raw = (ctypes.c_byte * self._n).from_address(ptr)
        itemsize = np.dtype(dtype).itemsize
        return np.frombuffer(raw, dtype=dtype,
                             count=self._n // itemsize).reshape(shape)

    def release(self) -> None:
        if self._h:
            self._lib.alloy_buffer_release(self._h)
            self._h = ctypes.c_void_p()


class Library:
    def __init__(self, ctx: Context, path: str) -> None:
        self._lib = ctx._lib
        self._h = ctypes.c_void_p()
        err = _Error()
        if self._lib.alloy_library_open(ctx._ctx, path.encode(),
                                        ctypes.byref(self._h),
                                        ctypes.byref(err)) != ALLOY_OK:
            raise AlloyUnavailable(
                f"opening the packaged ALLOY bundle: "
                f"{err.message.decode(errors='replace')}")

    def invoke(self, name: str, target: int, args) -> int:
        """Runs `name`; returns the target the runtime actually selected."""
        arr = (_Arg * len(args))(*args)
        res = _InvokeResult()
        err = _Error()
        if self._lib.alloy_invoke(self._h, name.encode(), target, arr, len(args),
                                  ctypes.byref(res),
                                  ctypes.byref(err)) != ALLOY_OK:
            raise AlloyUnavailable(
                f"ALLOY invoke {name}: {err.message.decode(errors='replace')}")
        return res.selected

    def close(self) -> None:
        if self._h:
            self._lib.alloy_library_close(self._h)
            self._h = ctypes.c_void_p()


def span(buf: Buffer, count: int) -> _Arg:
    return buf._lib.alloy_arg_span(buf._h, count)


def view2(buf: Buffer, nrows: int, ncols: int):
    """Returns (arg, desc). The desc must outlive the invoke, so it is returned."""
    desc = buf._lib.alloy_view2_contiguous(0, nrows, ncols)
    return buf._lib.alloy_arg_view2(buf._h, ctypes.byref(desc)), desc


def u64(v: int) -> _Arg:
    lib, _ = _bind()
    return lib.alloy_arg_u64(v)


def usize(v: int) -> _Arg:
    lib, _ = _bind()
    return lib.alloy_arg_usize(v)
