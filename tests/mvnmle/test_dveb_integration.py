"""Qualification tests for the branch-only DVEB MVN-MLE consumer adapter."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pystatistics.core.exceptions import DimensionError, ValidationError
from pystatistics.mvnmle import datasets, mlest
from pystatistics.mvnmle._dveb.loader import (
    ABI_VERSION,
    LIBRARY_PATH,
    LIBRARY_SHA256,
    SCHEDULE_AUTO,
    SCHEDULE_SERIAL,
    SCHEDULE_WORK_ITEM_PARALLEL,
)
from pystatistics.mvnmle._dveb.objective import DVEBDenseObjective


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def test_bundled_artifact_identity_and_dependencies():
    assert hashlib.sha256(LIBRARY_PATH.read_bytes()).hexdigest() == LIBRARY_SHA256
    output = subprocess.run(
        ["ldd", str(LIBRARY_PATH)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.lower()
    assert "libgomp" in output
    assert "torch" not in output
    assert "cuda" not in output
    assert "nvidia" not in output
    dynamic = subprocess.run(
        ["readelf", "-d", str(LIBRARY_PATH)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "RPATH" not in dynamic
    assert "RUNPATH" not in dynamic


@pytest.mark.skipif(not _torch_available(), reason="comparison needs PyTorch")
@pytest.mark.parametrize("data", [datasets.apple, datasets.missvals])
def test_objective_matches_incumbent_forward_cholesky(data):
    from pystatistics.mvnmle._objectives.gpu_fp64 import GPUObjectiveFP64

    incumbent = GPUObjectiveFP64(data, device="cpu")
    objective = DVEBDenseObjective(data, threads=1)
    theta = incumbent.get_initial_parameters()
    value_ref, gradient_ref = incumbent.compute_value_and_gradient(theta)
    value, gradient = objective.compute_value_and_gradient(theta)
    assert abs(value - value_ref) <= 1e-9 + 1e-9 * abs(value_ref)
    np.testing.assert_allclose(
        gradient,
        gradient_ref,
        atol=max(1e-9, 2e-8 * float(np.max(np.abs(gradient_ref)))),
        rtol=0.0,
    )
    objective.close()


def test_forced_and_automatic_schedules_preserve_mathematics():
    outputs = []
    for schedule in (
        SCHEDULE_AUTO,
        SCHEDULE_SERIAL,
        SCHEDULE_WORK_ITEM_PARALLEL,
    ):
        objective = DVEBDenseObjective(
            datasets.apple,
            threads=6,
            schedule=schedule,
        )
        outputs.append(objective.compute_value_and_gradient(objective.get_initial_parameters()))
        objective.close()
    for value_gradient in outputs[1:]:
        assert abs(value_gradient[0] - outputs[0][0]) <= 1e-12
        np.testing.assert_allclose(
            value_gradient[1], outputs[0][1], atol=1e-12, rtol=0.0
        )


def test_objective_boundary_refuses_implicit_copies_and_closed_context():
    objective = DVEBDenseObjective(datasets.apple, threads=1)
    theta = objective.get_initial_parameters()
    with pytest.raises(ValidationError, match="NumPy array"):
        objective.compute_value_and_gradient(theta.tolist())
    with pytest.raises(ValidationError, match="float64"):
        objective.compute_value_and_gradient(theta.astype(np.float32))
    with pytest.raises(DimensionError, match="shape"):
        objective.compute_value_and_gradient(theta[:-1])
    with pytest.raises(ValidationError, match="finite"):
        bad = theta.copy()
        bad[0] = np.nan
        objective.compute_value_and_gradient(bad)
    objective.close()
    objective.close()
    with pytest.raises(RuntimeError, match="closed"):
        objective.compute_value_and_gradient(theta)


def test_wrong_artifact_hash_refuses(tmp_path):
    bad = tmp_path / "wrong.so"
    bad.write_bytes(LIBRARY_PATH.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="integrity check"):
        DVEBDenseObjective(datasets.apple, library_path=bad)


def test_public_dveb_solver_matches_r_apple_reference():
    import json

    reference_path = Path(__file__).with_name("references") / "apple_reference.json"
    reference = json.loads(reference_path.read_text())
    result = mlest(datasets.apple, method="direct", backend="cpu", solver="dveb")
    assert result.converged
    assert result.backend_name == "cpu_dveb_cholesky_fp64"
    assert result.info["dveb_abi_version"] == ABI_VERSION
    assert result.info["dveb_artifact_sha256"] == LIBRARY_SHA256
    assert result.info["parameterization"] == "cholesky"
    assert abs(result.loglik - reference["loglik"]) < 1e-7
    np.testing.assert_allclose(result.muhat, reference["muhat"], rtol=1e-3)
    np.testing.assert_allclose(result.sigmahat, reference["sigmahat"], rtol=1e-3)


def test_default_cpu_route_remains_unchanged():
    if not _torch_available():
        pytest.skip("default fast route needs PyTorch")
    result = mlest(datasets.apple, backend="cpu")
    assert result.backend_name == "cpu_cholesky_fp64"


@pytest.mark.parametrize("method", ["em", "monotone"])
def test_dveb_solver_rejects_non_direct_methods(method):
    with pytest.raises(ValidationError, match="only valid with method='direct'"):
        mlest(datasets.apple, method=method, solver="dveb")


@pytest.mark.parametrize("backend", ["auto", "gpu", "gpu_fp64"])
def test_dveb_solver_rejects_non_cpu_backends(backend):
    with pytest.raises(ValidationError, match="CPU-only DVEB"):
        mlest(datasets.apple, method="direct", backend=backend, solver="dveb")


def test_dveb_solver_runs_when_torch_import_is_blocked():
    script = r'''
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch deliberately blocked by qualification")
        return None

sys.meta_path.insert(0, BlockTorch())
from pystatistics.mvnmle import datasets, mlest
result = mlest(datasets.apple, method="direct", backend="cpu", solver="dveb")
assert result.converged
assert result.backend_name == "cpu_dveb_cholesky_fp64"
assert "torch" not in sys.modules
print(result.loglik)
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[2])
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert np.isfinite(float(completed.stdout.strip()))
