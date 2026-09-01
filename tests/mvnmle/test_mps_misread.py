"""
Guards for the MPS allocator-state matmul misread on the MVNMLE GPU path.

An upstream torch bug (docs/GPU_NOTES.md, "MPS strided-matmul buffer
corruption") silently corrupts strided-matmul results at survey-scale
allocations on affected torch builds. MICE has been guarded against it since
6.1.3; MVNMLE was not, although it runs on the same backend, against the same
torch builds, and its FP32 objective forms ``sigma = L @ L.T`` -- whose second
operand is a transposed, therefore strided, view, which is exactly the shape
that misreads.

The defense is the same one MICE uses, and deliberately so: the library cannot
fix torch, so what it owes a user is honesty at the point of use. This module
pins that the warning fires when it should and stays quiet when it should not.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# Import scipy's BLAS before torch. Both link an OpenMP runtime, and on a
# macOS conda install loading torch first aborts the process with OMP error
# #15 (duplicate libomp). This is an environment property, not a project bug --
# but it makes the suite unrunnable on such a machine, and the documented
# workaround (KMP_DUPLICATE_LIB_OK) is disqualified here: it warns that it "may
# cause crashes or silently produce incorrect results", which is precisely the
# failure mode this module exists to guard against.
import scipy.linalg.blas  # noqa: F401  (import order matters; see above)

torch = pytest.importorskip("torch")

from pystatistics.mvnmle.backends import gpu as mvn_gpu


class TestMvnmleMisreadWarning:
    def test_warns_on_affected_mps_at_scale(self, monkeypatch):
        monkeypatch.setattr(torch, "__version__", "2.12.1")
        with pytest.warns(UserWarning, match="silently wrong GPU results"):
            mvn_gpu._warn_if_mps_misread("mps", 50_000, 50)

    def test_the_warning_names_every_remedy(self, monkeypatch):
        # A warning that says only "results may be wrong" leaves the reader
        # with nothing to do about it.
        monkeypatch.setattr(torch, "__version__", "2.12.1")
        with pytest.warns(UserWarning) as rec:
            mvn_gpu._warn_if_mps_misread("mps", 50_000, 50)
        text = str(rec[0].message)
        assert "backend='cpu'" in text
        assert "CUDA" in text
        assert "2.14" in text
        assert "mps_matmul_canary" in text

    def test_the_warning_names_the_operation_that_misreads(self, monkeypatch):
        # The point of naming `L @ L.T` is that a reader can check whether the
        # claim applies to their own build rather than take it on faith.
        monkeypatch.setattr(torch, "__version__", "2.12.1")
        with pytest.warns(UserWarning) as rec:
            mvn_gpu._warn_if_mps_misread("mps", 50_000, 50)
        assert "L @ L.T" in str(rec[0].message)

    def test_silent_on_mitigated_torch(self, monkeypatch):
        # Installing torch >= 2.14 is the supported opt-in path to silence it.
        monkeypatch.setattr(torch, "__version__", "2.14.0")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mvn_gpu._warn_if_mps_misread("mps", 50_000, 50)

    def test_silent_below_threshold(self, monkeypatch):
        monkeypatch.setattr(torch, "__version__", "2.12.1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mvn_gpu._warn_if_mps_misread("mps", 100, 5)

    def test_silent_on_cuda_and_cpu(self, monkeypatch):
        # The defect is Metal-only; warning on CUDA would be noise that trains
        # users to ignore it.
        monkeypatch.setattr(torch, "__version__", "2.12.1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mvn_gpu._warn_if_mps_misread("cuda", 50_000, 50)
            mvn_gpu._warn_if_mps_misread("cpu", 50_000, 50)

    def test_threshold_matches_mice(self):
        # One number, not two: the defect is a property of the torch build and
        # the allocation history, not of which estimator is running, so the two
        # guards must not drift apart.
        from pystatistics.mice.backends import gpu as mice_gpu

        assert mvn_gpu._MPS_MISREAD_WARN_N == mice_gpu._MPS_MISREAD_WARN_N


class TestReachedFromARealSolve:
    """The guard is worth nothing if `solve` does not call it."""

    def test_solve_consults_the_guard(self, monkeypatch):
        from pystatistics.mvnmle.design import MVNDesign

        seen: list[tuple] = []
        monkeypatch.setattr(
            mvn_gpu, "_warn_if_mps_misread", lambda d, n, p: seen.append((d, n, p))
        )
        rng = np.random.default_rng(0)
        data = rng.normal(size=(40, 3))
        data[0, 0] = np.nan
        design = MVNDesign.from_array(data)

        backend = mvn_gpu.DirectMLEBackend(device="cpu", use_fp64=True)
        backend.solve(design)

        assert seen, "DirectMLEBackend.solve must consult the MPS misread guard"
        assert seen[0][0] == "cpu"
        assert seen[0][1] == design.n
        assert seen[0][2] == design.p
