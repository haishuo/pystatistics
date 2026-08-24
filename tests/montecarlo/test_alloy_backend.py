"""The packaged ALLOY bootstrap backend, reached through the public API.

Every test here calls `boot(...)` -- the point of the slice is that an existing
public call routes through ALLOY, so testing the backend class directly would
check the wrong thing.
"""

import pathlib
import platform
import sys

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.montecarlo import boot

APPLE = sys.platform == "darwin" and platform.machine() == "arm64"


def mean_stat(data, indices):
    return np.array([np.mean(data[indices])])


def median_stat(data, indices):
    return np.array([np.median(data[indices])])


@pytest.fixture(scope="module")
def alloy():
    if not APPLE:
        pytest.skip("the packaged ALLOY backend is macOS/arm64 only")
    from pystatistics.montecarlo.backends import _alloy
    ok, why = _alloy.is_available()
    if not ok:
        pytest.skip(f"ALLOY unavailable: {why}")
    return _alloy


# ------------------------------------------------------------ gate 1 and 2

def test_the_public_call_routes_through_alloy_metal(alloy):
    """`backend_name` DISCLOSES the implementation, and it is not PyTorch."""
    data = np.arange(1.0, 51.0)
    r = boot(data, mean_stat, n_resamples=1000, seed=42, backend="gpu",
             gpu_statistic="mean")
    assert r.backend_name == "gpu_mps_alloy_bootstrap"
    # The naming convention the rest of the library's tests rely on.
    assert "gpu" in r.backend_name and "bootstrap" in r.backend_name
    info = r._result.info
    assert info["implementation"] == "alloy"
    assert len(info["alloy_commit"]) == 40
    assert r.t.shape == (1000, 1)


def test_alloy_does_the_resampling_and_the_averaging(alloy):
    """Not merely reached: the replicate means came out of the artifact.

    The kernels draw every index and compute every replicate mean. If Python
    were doing either, the values would be f64 throughout; they are f32
    computed on Metal and widened, so they land exactly on f32 values.
    """
    data = np.arange(1.0, 51.0)
    r = boot(data, mean_stat, n_resamples=500, seed=7, backend="gpu",
             gpu_statistic="mean")
    t = r.t[:, 0]
    assert np.array_equal(t, t.astype(np.float32).astype(np.float64))
    # Every replicate mean is inside the sample's own range, which is what
    # resampling with replacement guarantees and a broken index would break.
    assert t.min() >= data.min() and t.max() <= data.max()


# ---------------------------------------------------------------- gate 3

def test_a_fixed_seed_reproduces_exactly(alloy):
    data = np.arange(1.0, 31.0)
    a = boot(data, mean_stat, n_resamples=400, seed=123, backend="gpu",
             gpu_statistic="mean")
    b = boot(data, mean_stat, n_resamples=400, seed=123, backend="gpu",
             gpu_statistic="mean")
    np.testing.assert_array_equal(a.t, b.t)
    assert a.standard_errors == b.standard_errors


def test_different_seeds_differ(alloy):
    data = np.arange(1.0, 31.0)
    a = boot(data, mean_stat, n_resamples=400, seed=1, backend="gpu",
             gpu_statistic="mean")
    b = boot(data, mean_stat, n_resamples=400, seed=2, backend="gpu",
             gpu_statistic="mean")
    assert not np.array_equal(a.t, b.t)


# ---------------------------------------------------------------- gate 4

def test_the_statistical_contract_holds(alloy):
    """t0 exact, SE near the analytic value, bias near zero."""
    rng = np.random.default_rng(0)
    data = rng.normal(10.0, 2.0, 400)
    r = boot(data, mean_stat, n_resamples=4000, seed=99, backend="gpu",
             gpu_statistic="mean")
    assert r.t0[0] == pytest.approx(float(np.mean(data)), rel=1e-12)
    analytic_se = float(np.std(data, ddof=1)) / np.sqrt(len(data))
    assert r.standard_errors[0] == pytest.approx(analytic_se, rel=0.10)
    assert abs(r.bias[0]) < 4.0 * analytic_se


def test_it_agrees_with_the_cpu_backend_statistically(alloy):
    """Independent streams, so t0 is identical and SE agrees within MC error."""
    data = np.arange(1.0, 51.0)
    cpu = boot(data, mean_stat, n_resamples=4000, seed=42, backend="cpu")
    gpu = boot(data, mean_stat, n_resamples=4000, seed=42, backend="gpu",
               gpu_statistic="mean")
    np.testing.assert_array_equal(gpu.t0, cpu.t0)
    np.testing.assert_allclose(gpu.standard_errors, cpu.standard_errors,
                               rtol=0.15)


# ---------------------------------------------------------------- gate 5

def test_the_declaration_guard_still_refuses_a_non_mean(alloy):
    data = np.random.default_rng(0).gamma(2, 3, 300)
    with pytest.raises(ValidationError, match="does not equal mean"):
        boot(data, median_stat, n_resamples=99, seed=1, backend="gpu",
             gpu_statistic="mean")


@pytest.mark.parametrize("kwargs", [
    {"method": "balanced"},
    {"statistic_type": "frequency"},
])
def test_unsupported_configurations_are_still_refused(alloy, kwargs):
    """The ALLOY path did not widen what the GPU accepts."""
    data = np.arange(1.0, 21.0)
    with pytest.raises(ValidationError):
        boot(data, mean_stat, n_resamples=50, seed=1, backend="gpu",
             gpu_statistic="mean", **kwargs)


def test_strata_and_2d_data_are_still_refused(alloy):
    data = np.arange(1.0, 21.0)
    with pytest.raises(ValidationError):
        boot(data, mean_stat, n_resamples=50, seed=1, backend="gpu",
             gpu_statistic="mean", strata=np.repeat([0, 1], 10))
    with pytest.raises(ValidationError):
        boot(data.reshape(10, 2), mean_stat, n_resamples=50, seed=1,
             backend="gpu", gpu_statistic="mean")


def test_gpu_without_a_declaration_still_raises(alloy):
    data = np.arange(1.0, 21.0)
    with pytest.raises(ValidationError, match="requires gpu_statistic='mean'"):
        boot(data, mean_stat, n_resamples=50, seed=1, backend="gpu")


# ---------------------------------------------------------------- gate 7

def test_the_artifacts_are_loaded_only_from_the_package(alloy):
    """No source tree, no build directory, no environment variable.

    A numerical result whose provenance depends on which machine ran it is not
    reproducible, so the loader has exactly one place to look. This asserts the
    path it used is inside the installed package.
    """
    from pathlib import Path

    import pystatistics.montecarlo.backends._alloy as mod
    pkg = Path(mod.__file__).resolve().parent
    assert mod._ARTIFACTS == pkg / "artifacts"
    assert (mod._ARTIFACTS / "liballoy.dylib").is_file()
    assert (mod._ARTIFACTS / "bootstrap.alloylib" / "manifest.json").is_file()


def test_the_provenance_record_matches_the_shipped_binaries(alloy):
    """Every digest checked, which is what makes a partial refresh fail loudly."""
    import hashlib

    rec = alloy.provenance()
    assert rec["platform"] == {"system": "Darwin", "machine": "arm64"}
    assert rec["bundle_format_version"].startswith("0.10.")
    for rel, want in rec["artifacts"].items():
        got = hashlib.sha256((alloy._ARTIFACTS / rel).read_bytes()).hexdigest()
        assert got == want, rel
    for fn in ("resample_partials", "reduce_partials"):
        assert fn in rec["exported_functions"]


def test_a_corrupt_artifact_is_refused_by_name(alloy, tmp_path, monkeypatch):
    """The failure mode this guard exists for, exercised rather than asserted."""
    rec = dict(alloy.provenance())
    rec["artifacts"] = dict(rec["artifacts"])
    rec["artifacts"]["liballoy.dylib"] = "0" * 64
    with pytest.raises(alloy.AlloyUnavailable, match="does not match its recorded"):
        alloy._verify(rec)

    missing = dict(rec)
    missing["artifacts"] = {"nope.dylib": "0" * 64}
    with pytest.raises(alloy.AlloyUnavailable, match="is missing"):
        alloy._verify(missing)


# ---------------------------------------------------------------- gate 8

def test_the_cpu_path_is_untouched():
    """Runs everywhere, including where ALLOY is unavailable."""
    data = np.arange(1.0, 21.0)
    r = boot(data, mean_stat, n_resamples=200, seed=5, backend="cpu")
    assert "cpu" in r.backend_name
    assert r.t0[0] == pytest.approx(10.5)


def test_auto_does_not_reach_for_metal():
    """`auto` resolves to the CPU on Apple Silicon, exactly as before.

    The convention is that `auto` never selects MPS. Routing the *explicit*
    GPU request to ALLOY must not have changed that, or `auto` would silently
    start producing a different (f32) answer.
    """
    data = np.arange(1.0, 51.0)
    r = boot(data, mean_stat, n_resamples=200, seed=42, backend="auto",
             gpu_statistic="mean")
    if APPLE:
        assert "cpu" in r.backend_name


# ------------------------------------------------- shared-session lifecycle

def test_closing_a_backend_does_not_disable_the_next_call(alloy):
    """The regression: `close()` used to shut the process-wide session.

    A backend instance is created per solve, so one caller invoking `close()`
    left the cache holding a closed context and broke every subsequent
    `boot(...)` on the machine. Closing must not reach the shared session.
    """
    from pystatistics.montecarlo.backends.alloy import ALLOYBootstrapBackend

    data = np.arange(1.0, 41.0)
    first = boot(data, mean_stat, n_resamples=200, seed=3, backend="gpu",
                 gpu_statistic="mean")

    be = ALLOYBootstrapBackend()
    be.close()
    be.close()          # idempotent, and still not the session's business

    second = boot(data, mean_stat, n_resamples=200, seed=3, backend="gpu",
                  gpu_statistic="mean")
    assert second.backend_name == "gpu_mps_alloy_bootstrap"
    np.testing.assert_array_equal(first.t, second.t)


def test_the_session_is_shared_rather_than_rebuilt(alloy):
    """Two backends hand back the same context and library objects."""
    from pystatistics.montecarlo.backends.alloy import ALLOYBootstrapBackend

    a, b = ALLOYBootstrapBackend(), ALLOYBootstrapBackend()
    assert a._ctx is b._ctx
    assert a._lib is b._lib


# --------------------------------------------- routing without PyTorch

def test_the_eligible_gpu_path_does_not_import_torch(alloy, monkeypatch):
    """Selection must not reach for PyTorch merely to learn Metal exists.

    Simulated by making `import torch` fail: the eligible explicit-GPU request
    must still select ALLOY. `torchfree_check.py` beside this file is the real
    proof -- a wheel installed with no torch at all -- and this is the fast
    check that keeps the selection order from regressing.
    """
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    data = np.arange(1.0, 51.0)
    r = boot(data, mean_stat, n_resamples=300, seed=42, backend="gpu",
             gpu_statistic="mean")
    assert r.backend_name == "gpu_mps_alloy_bootstrap"


@pytest.fixture
def no_torch(monkeypatch):
    """Make `import torch` fail, so a path that needs it is caught here."""
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    return guarded


def test_a_missing_declaration_still_names_the_declaration(alloy, no_torch):
    """The regression: this answered "No GPU available ... install PyTorch".

    An unsupported CONFIGURATION is not a missing device. The packaged runtime
    has already established that Metal is here, so falling through to a
    resolver that asks torch produced a wrong diagnosis, not merely an
    unhelpful one.
    """
    data = np.arange(1.0, 51.0)
    with pytest.raises(ValidationError, match="requires gpu_statistic='mean'"):
        boot(data, mean_stat, n_resamples=50, seed=1, backend="gpu")


@pytest.mark.parametrize("kwargs", [
    {"method": "balanced"},
    {"method": "parametric", "ran_gen": lambda d, m, r: d},
    {"statistic_type": "frequency"},
    {"statistic_type": "weight"},
])
def test_unsupported_configurations_still_name_the_configuration(alloy, no_torch,
                                                                 kwargs):
    data = np.arange(1.0, 21.0)
    with pytest.raises(ValidationError, match="supports only"):
        boot(data, mean_stat, n_resamples=50, seed=1, backend="gpu",
             gpu_statistic="mean", **kwargs)


def test_strata_and_2d_still_name_the_configuration(alloy, no_torch):
    data = np.arange(1.0, 21.0)
    with pytest.raises(ValidationError, match="supports only"):
        boot(data, mean_stat, n_resamples=50, seed=1, backend="gpu",
             gpu_statistic="mean", strata=np.repeat([0, 1], 10))
    with pytest.raises(ValidationError, match="supports only"):
        boot(data.reshape(10, 2), mean_stat, n_resamples=50, seed=1,
             backend="gpu", gpu_statistic="mean")


def test_the_refusal_wording_is_one_copy(alloy):
    """Both callers of the refusal must raise the same words.

    The Apple-Silicon path reaches it without the resolver and every other
    device reaches it through the resolver; a fork would drift silently,
    because each side has tests that only ever see its own copy.
    """
    from pystatistics.montecarlo import solvers

    src = pathlib.Path(solvers.__file__).read_text()
    assert src.count("requires gpu_statistic='mean'. The GPU bootstrap") == 1
    assert src.count("supports only\n        \"method='ordinary'") == 0
    assert src.count("_refuse_gpu_boot_design(design)") == 2


def test_an_unavailable_alloy_fails_loudly_rather_than_substituting(alloy,
                                                                    monkeypatch):
    """No silent PyTorch, no silent CPU -- the ALLOY reason is what surfaces."""
    from pystatistics.montecarlo.backends import _alloy as mod

    monkeypatch.setattr(mod, "is_available",
                        lambda: (False, "no usable Metal device (simulated)."))
    data = np.arange(1.0, 51.0)
    with pytest.raises(mod.AlloyUnavailable, match="simulated"):
        boot(data, mean_stat, n_resamples=100, seed=1, backend="gpu",
             gpu_statistic="mean")
