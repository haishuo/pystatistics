"""Apple-Silicon bootstrap of the mean, computed by a packaged ALLOY artifact.

Reached only through :func:`pystatistics.montecarlo.solvers._select_boot_backend`,
on exactly the designs it already routes to the GPU: a caller-declared
``gpu_statistic='mean'``, ordinary method, index-type statistic, no strata, 1-D
data. This backend narrows that further to a Metal device -- CUDA keeps the
PyTorch path unchanged.

WHAT ALLOY DOES HERE. Both halves of the bootstrap: the resample indices and the
replicate means. ``resample_partials`` draws every index and sums the sampled
values per (replicate, tile); ``reduce_partials`` sums the tiles and divides.
Python allocates, computes ``t0`` with the caller's own statistic, verifies the
declaration, and builds the result object -- it performs no resampling and no
averaging.

The random stream is a pure function of ``(seed, replicate, draw)``, so a fixed
seed reproduces exactly, and the drawn indices are identical on the CPU and on
Metal. That is stronger than the PyTorch path, which documents its GPU stream as
statistically equivalent to the CPU's but not identical.
"""

from __future__ import annotations

import numpy as np

from pystatistics.core.compute.timing import Timer
from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import Result
from pystatistics.montecarlo._common import BootParams
from pystatistics.montecarlo.backends import _alloy
from pystatistics.montecarlo.design import BootstrapDesign

#: Relative tolerance for verifying a declared GPU statistic against the value
#: observed on the original data. Kept identical to the PyTorch backend's: the
#: guard is a property of the declaration, not of who computes the resamples.
_DECLARATION_RTOL = 1e-9


def _tiles(n: int, n_resamples: int) -> int:
    """How many tiles one replicate's draws are split across.

    One rule, not a table of measured optima. Two ceilings, whichever binds:
    at least ~16 draws per thread so the per-thread setup is amortised, and a
    bounded partials matrix because a second pass has to read it back. Rounded
    down to a power of two -- the measured curve is flat over a wide middle,
    but non-power-of-two tile counts are consistently worse.
    """
    t = min(max(1, n // 16), max(1, 524_288 // n_resamples), 128)
    p = 1
    while p * 2 <= t:
        p *= 2
    return p


class ALLOYBootstrapBackend:
    """Vectorized bootstrap of a declared-mean statistic, on ALLOY/Metal."""

    def __init__(self) -> None:
        # The session is process-wide (see _alloy.shared_session): the bundle is
        # immutable package data, and re-opening it per call cost 5.7 ms.
        self._ctx, self._lib = _alloy.shared_session()
        if not self._ctx.has_gpu:
            raise _alloy.AlloyUnavailable(
                f"no usable Metal device ({self._ctx.device_name})")
        self._device = self._ctx.device_name

    @property
    def name(self) -> str:
        # Contains 'gpu' and 'bootstrap', which the library's backend-naming
        # tests require, and discloses ALLOY, which the caller needs in order
        # to know which implementation produced the numbers.
        return "gpu_mps_alloy_bootstrap"

    def solve(self, design: BootstrapDesign) -> Result[BootParams]:
        timer = Timer()
        timer.start()

        data = np.ascontiguousarray(design.data, dtype=np.float64)
        n = data.shape[0]
        n_resamples = design.n_resamples
        seed = design.seed if design.seed is not None else 0

        # t0 from the CALLER'S OWN function, in f64 on the CPU, exactly as the
        # PyTorch backend does. The GPU path never infers the statistic.
        with timer.section("t0_computation"):
            t0 = np.atleast_1d(np.asarray(
                design.statistic(design.data, np.arange(n)), dtype=np.float64))
            k = len(t0)

        with timer.section("declaration_check"):
            mean_all = float(np.mean(data))
            if k != 1 or abs(t0[0] - mean_all) > _DECLARATION_RTOL * (
                abs(mean_all) + _DECLARATION_RTOL
            ):
                raise ValidationError(
                    "gpu_statistic='mean' was declared, but statistic(data, "
                    "all-indices) does not equal mean(data) "
                    f"(got {t0.tolist()}, expected {mean_all}). The GPU path "
                    "computes the sample mean; refusing to silently compute a "
                    "different quantity. Use backend='cpu' for this statistic."
                )

        tiles = _tiles(n, n_resamples)
        lib = self._lib
        db = pb = ob = None
        try:
            with timer.section("gpu_compute"), _alloy.invoke_lock:
                db = self._ctx.buffer(n * 4)
                pb = self._ctx.buffer(n_resamples * tiles * 4)
                ob = self._ctx.buffer(n_resamples * 4)
                db.as_np(np.float32, (n,))[:] = data.astype(np.float32)

                part, _d1 = _alloy.view2(pb, n_resamples, tiles)
                got = lib.invoke("resample_partials", _alloy.TARGET_GPU,
                                 [_alloy.u64(seed), _alloy.span(db, n), part])
                part2, _d2 = _alloy.view2(pb, n_resamples, tiles)
                got2 = lib.invoke("reduce_partials", _alloy.TARGET_GPU,
                                  [part2, _alloy.span(ob, n_resamples),
                                   _alloy.usize(n)])
                # NO SILENT FALLBACK: the runtime reports which target it
                # selected, and anything but Metal is an error rather than a
                # quietly different implementation.
                if got != _alloy.TARGET_GPU or got2 != _alloy.TARGET_GPU:
                    raise _alloy.AlloyUnavailable(
                        "an ALLOY bootstrap phase did not run on Metal "
                        f"(targets {got}, {got2})")
                t = ob.as_np(np.float32, (n_resamples,)).astype(
                    np.float64).reshape(n_resamples, 1)
        finally:
            for b in (db, pb, ob):
                if b is not None:
                    b.release()

        with timer.section("summary_statistics"):
            bias = np.mean(t, axis=0) - t0
            standard_errors = np.std(t, axis=0, ddof=1)

        timer.stop()
        return Result(
            params=BootParams(
                t0=t0, t=t, n_resamples=n_resamples, bias=bias,
                standard_errors=standard_errors, conf_int=None, conf_level=None,
            ),
            info={
                "method": design.method,
                "statistic_type": design.statistic_type,
                "n": n, "k": k, "gpu_vectorized": True,
                "implementation": "alloy",
                "alloy_commit": _alloy.provenance()["alloy_commit"],
                "alloy_tiles": tiles,
                "device": self._device,
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=(),
        )

    def close(self) -> None:
        self._ctx.close()
