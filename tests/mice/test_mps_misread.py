"""
Guards for the MPS allocator-state matmul-misread defenses (6.1.3).

An upstream torch bug (docs/GPU_NOTES.md, "MPS strided-matmul buffer
corruption") silently corrupts strided-matmul results at survey-scale
allocations on affected torch builds — measured as ~1 wrong-but-finite chain
fit in 20 at n=50k, invisible to every non-finite guard. The library cannot
fix torch, so the defense is honesty at the point of use: a version
classifier (``core.compute.device.mps_misread_status``), a loud UserWarning
before survey-scale MPS runs on affected builds, and an empirical canary
(``mice.diagnostics.mps_matmul_canary``). This module pins all three.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pystatistics.core.compute.device import mps_misread_status
from pystatistics.mice import mice
from pystatistics.mice.backends import gpu as gpu_backend
from pystatistics.mice.design import MICEDesign


def _has_mps() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


class TestMisreadClassifier:
    """The version table encodes exactly what the 2026-08-13 bisect verified:
    everything through 2.13.x corrupts; the mitigation entered the 2.14
    nightlies at dev20260624 (pytorch #187441 landed 2026-06-23; dev20260622
    corrupts, dev20260624 is clean) and is in the 2.14 release branch."""

    @pytest.mark.parametrize(
        "version, expected",
        [
            ("2.7.1", "affected"),
            ("2.9.0", "affected"),
            ("2.12.1", "affected"),
            ("2.13.0", "affected"),
            # A 2.13.x patch release without a verified cherry-pick must stay
            # conservative: the guarded failure is silent wrong numbers.
            ("2.13.1", "affected"),
            ("2.14.0.dev20260622", "affected"),
            ("2.14.0.dev20260624", "mitigated"),
            ("2.14.0.dev20260801", "mitigated"),
            # Release-branch builds (rc / a0+git) contain #187441 (branch cut
            # 2026-08-11).
            ("2.14.0", "mitigated"),
            ("2.14.0rc1", "mitigated"),
            ("2.14.1", "mitigated"),
            ("2.15.0.dev20260813", "mitigated"),
            ("2.15.0", "mitigated"),
            ("2.16.0.dev20270101", "mitigated"),
            # Unparseable: warn rather than trust (fail loud, Rule 1).
            ("garbage", "affected"),
        ],
    )
    def test_version_table(self, monkeypatch, version, expected):
        monkeypatch.setattr(torch, "__version__", version)
        assert mps_misread_status() == expected

    def test_matches_installed_torch(self):
        # Whatever is installed must classify (no exception, valid label).
        assert mps_misread_status() in ("affected", "mitigated")


class TestMisreadWarning:
    """`_warn_if_mps_misread` must warn exactly when it should: MPS device,
    survey-scale n, affected torch — and stay silent otherwise, so installing
    a mitigated torch is the supported opt-in path to a quiet run."""

    def test_warns_on_affected_mps_at_scale(self, monkeypatch):
        monkeypatch.setattr(torch, "__version__", "2.12.1")
        with pytest.warns(UserWarning, match="silently wrong"):
            gpu_backend._warn_if_mps_misread("mps", 50_000)

    def test_silent_on_mitigated_torch(self, monkeypatch):
        monkeypatch.setattr(torch, "__version__", "2.14.0")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gpu_backend._warn_if_mps_misread("mps", 50_000)

    def test_silent_below_threshold(self, monkeypatch):
        monkeypatch.setattr(torch, "__version__", "2.12.1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gpu_backend._warn_if_mps_misread(
                "mps", gpu_backend._MPS_MISREAD_WARN_N - 1
            )

    def test_silent_on_cuda(self, monkeypatch):
        monkeypatch.setattr(torch, "__version__", "2.12.1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gpu_backend._warn_if_mps_misread("cuda", 50_000)

    @pytest.mark.skipif(not _has_mps(), reason="needs MPS")
    def test_fires_from_real_mice_run(self, monkeypatch):
        """The call site is wired: a real (tiny, threshold-lowered) MPS mice
        run warns iff the installed torch is affected."""
        monkeypatch.setattr(gpu_backend, "_MPS_MISREAD_WARN_N", 50)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 3))
        X[rng.random(200) < 0.2, 0] = np.nan
        design = MICEDesign.from_array(X)
        if mps_misread_status() == "affected":
            with pytest.warns(UserWarning, match="silently wrong"):
                mice(design, n_imputations=2, max_iter=2, seed=0, backend="gpu")
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                mice(design, n_imputations=2, max_iter=2, seed=0, backend="gpu")
            assert not [w for w in caught if "silently wrong" in str(w.message)]


class TestCanary:
    """The canary must produce an unambiguous, well-formed verdict. Its
    discrimination was measured at 6 orders of magnitude (deviation 42.9 on
    an affected build vs 1.1e-5 on a mitigated one), so any drift into the
    threshold's neighbourhood is itself a finding."""

    @pytest.mark.skipif(not _has_mps(), reason="needs MPS")
    def test_canary_runs_and_verdict_is_sharp(self):
        from pystatistics.mice.diagnostics import (
            _CANARY_DEVIATION_LIMIT,
            mps_matmul_canary,
        )

        r = mps_matmul_canary()
        assert r["status"] in ("clean", "corrupted")
        assert r["n"] == 50_000 and r["m"] == 20
        assert len(r["per_chain_deviation"]) == r["m"]
        assert r["max_deviation"] == max(r["per_chain_deviation"])
        # Sharpness: the verdict must not sit near the threshold.
        if r["status"] == "clean":
            assert r["max_deviation"] < 0.1 * _CANARY_DEVIATION_LIMIT
        else:
            assert r["max_deviation"] > 5.0 * _CANARY_DEVIATION_LIMIT

    def test_canary_raises_without_mps(self, monkeypatch):
        from pystatistics.mice import diagnostics

        if _has_mps():
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="MPS"):
            diagnostics.mps_matmul_canary()
