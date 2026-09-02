"""Qualification tests for the private DVEB exact-ARMA consumer adapter."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pystatistics.core.exceptions import DimensionError, ValidationError
from pystatistics.timeseries import arima_batch
from pystatistics.timeseries._dveb_arma import (
    DVEBCPUExactArma,
    DVEBCudaTransferExactArma,
)
from pystatistics.timeseries._dveb_arma.loader import (
    ABI_VERSION,
    CPU_ITEM_PARALLEL,
    CPU_LIBRARY_PATH,
    CPU_LIBRARY_SHA256,
    CPU_SERIAL,
    MANIFEST,
    _missing_x86_64_v2_features,
)

COMBINED = Path("/mnt/artifacts/dveb/trunk009_exact_arma_20260902/exact_arma_abi_v1.so")


def inputs(items=8, steps=120, state=1):
    rng = np.random.Generator(np.random.PCG64DXSM(0x445645425F434F4E))
    return (
        np.ascontiguousarray(rng.standard_normal((items, steps))),
        np.full((items, state), 0.6 if state == 1 else 0.0, dtype=np.float64),
        np.pad(np.ones((items, 1), dtype=np.float64), ((0, 0), (0, state - 1))),
    )


def test_cpu_artifact_identity_and_dependencies():
    assert MANIFEST["abi_version"] == ABI_VERSION
    assert MANIFEST["cpu_isa"] == "x86-64-v2"
    assert not MANIFEST["cpu_cuda_runtime_dependency"]
    assert hashlib.sha256(CPU_LIBRARY_PATH.read_bytes()).hexdigest() == CPU_LIBRARY_SHA256
    output = subprocess.run(
        ["readelf", "-d", str(CPU_LIBRARY_PATH)], check=True,
        capture_output=True, text=True,
    ).stdout.lower()
    assert "cuda" not in output
    assert "nvidia" not in output
    assert "rpath" not in output
    assert "runpath" not in output


def test_x86_64_v2_feature_check():
    flags = frozenset({"cx16", "lahf_lm", "popcnt", "pni", "ssse3", "sse4_1", "sse4_2"})
    assert not _missing_x86_64_v2_features(flags)
    assert _missing_x86_64_v2_features(flags - {"popcnt"}) == {"popcnt"}


def test_cpu_forced_schedules_match_and_leave_inputs_unchanged():
    values = inputs()
    before = [value.copy() for value in values]
    answers = []
    for schedule, threads in ((CPU_SERIAL, 1), (CPU_ITEM_PARALLEL, 6)):
        with DVEBCPUExactArma(max_threads=6, schedule=schedule) as evaluator:
            answers.append(evaluator.evaluate(*values, threads=threads))
            assert evaluator.last_selected_schedule == schedule
            assert evaluator.scratch_bytes > 0
    assert all(np.array_equal(left, right) for left, right in zip(answers[0], answers[1]))
    assert all(np.array_equal(left, right) for left, right in zip(values, before))


def test_cpu_refuses_auto_bad_inputs_and_closed_context(tmp_path):
    with pytest.raises(ValidationError, match="automatic selector"):
        DVEBCPUExactArma(max_threads=6, schedule=0)
    with pytest.raises(RuntimeError, match="missing.*No fallback"):
        DVEBCPUExactArma(max_threads=1, schedule=CPU_SERIAL, library_path=tmp_path / "none.so")
    bad = tmp_path / "bad.so"
    bad.write_bytes(CPU_LIBRARY_PATH.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="integrity"):
        DVEBCPUExactArma(max_threads=1, schedule=CPU_SERIAL, library_path=bad)

    evaluator = DVEBCPUExactArma(max_threads=2, schedule=CPU_SERIAL)
    z, phi, loading = inputs()
    with pytest.raises(ValidationError, match="NumPy"):
        evaluator.evaluate(z.tolist(), phi, loading, threads=1)
    with pytest.raises(ValidationError, match="float64"):
        evaluator.evaluate(z.astype(np.float32), phi, loading, threads=1)
    with pytest.raises(ValidationError, match="C-contiguous"):
        evaluator.evaluate(z[:, ::2], phi, loading, threads=1)
    with pytest.raises(DimensionError, match="batch shapes"):
        evaluator.evaluate(z, phi[:-1], loading[:-1], threads=1)
    with pytest.raises(ValidationError, match="threads"):
        evaluator.evaluate(z, phi, loading, threads=3)
    evaluator.close()
    evaluator.close()
    with pytest.raises(RuntimeError, match="closed"):
        evaluator.evaluate(z, phi, loading, threads=1)


def test_cpu_adapter_runs_with_torch_import_blocked():
    script = r'''
import importlib.abc
import sys
import numpy as np
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch deliberately blocked")
        return None
sys.meta_path.insert(0, BlockTorch())
from pystatistics.timeseries._dveb_arma import DVEBCPUExactArma
z = np.ones((2, 16), dtype=np.float64)
phi = np.full((2, 1), 0.6, dtype=np.float64)
loading = np.ones((2, 1), dtype=np.float64)
with DVEBCPUExactArma(max_threads=1, schedule=1) as evaluator:
    nll, sigma2, status = evaluator.evaluate(z, phi, loading, threads=1)
assert status.all() and np.isfinite(nll).all() and np.isfinite(sigma2).all()
assert "torch" not in sys.modules
print("PASS")
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[2])
    completed = subprocess.run(
        [sys.executable, "-c", script], env=env, check=True,
        capture_output=True, text=True,
    )
    assert completed.stdout.strip() == "PASS"


def test_cuda_requires_explicit_artifact_and_refuses_bad_block():
    with pytest.raises(RuntimeError, match="not bundled"):
        DVEBCudaTransferExactArma(max_items=8, max_steps=120, max_state=1, library_path=None)
    if not COMBINED.is_file():
        pytest.skip("qualified external CUDA artifact unavailable")
    try:
        evaluator = DVEBCudaTransferExactArma(
            max_items=8, max_steps=120, max_state=1, library_path=COMBINED
        )
    except RuntimeError as exc:
        if "CUDA" in str(exc):
            pytest.skip(str(exc))
        raise
    try:
        z, phi, loading = inputs()
        forced = evaluator.evaluate(z, phi, loading, block=32)
        assert evaluator.last_selected_block == 32
        automatic = evaluator.evaluate(z, phi, loading, block=0)
        assert evaluator.last_selected_block == 32
        assert all(np.array_equal(left, right) for left, right in zip(forced, automatic))
        with pytest.raises(ValidationError, match="block override"):
            evaluator.evaluate(z, phi, loading, block=16)
        assert evaluator.payload_bytes > 0
    finally:
        evaluator.close()


def test_public_arima_batch_contract_is_unchanged():
    with pytest.raises(ValidationError, match="only method='whittle'"):
        arima_batch(np.ones((2, 32)), order=(1, 0, 0), method="ml")
