"""Reuse frozen MVN-MLE cases without adding the checkout to package imports."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "dveb_mvnmle_integration" / "common.py"
_SPEC = importlib.util.spec_from_file_location("_dveb_frozen_common", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load frozen common module at {_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

CASE_IDS = _MODULE.CASE_IDS
THREADS = _MODULE.THREADS
WARMUPS = _MODULE.WARMUPS
REPETITIONS = _MODULE.REPETITIONS
CASES = _MODULE.CASES
affinity = _MODULE.affinity
array_sha256 = _MODULE.array_sha256
compare_solutions = _MODULE.compare_solutions
configure_threads = _MODULE.configure_threads
solution_record = _MODULE.solution_record

SEED = 0xD0EB1009
EXPECTED_ARTIFACT_SHA256 = (
    "65438a0d742257acc0dc7bd2ff13d2017669fd21dc07f64add41b18fb229fe38"
)
FIXTURES = Path(__file__).with_name("fixtures")
EXPECTED_INPUT_SHA256 = {
    "E1": "bd95a4303d6cf1246c74a3b65e87b1e9e56d3a559f814c58955d8265ad3ff6be",
    "E2": "8d1890a935ae6c87451f035a9508cd12d06418aa4e5d9b00d716dc5ff57a4817",
    "E3": "8046f7cd997e4340c2db405fda4e4721fa46f873514d46aebff5fd8795d5d5e0",
    "E4": "05cb9bfa4bf37d46cc2cc4c31b33ec99b963eac51fa00d09e12d85b9e9af580e",
    "E5": "e649b5f23e6f9f7cb413b1f79578d96744c224bc20bd73e5dd449e5220db6da4",
    "E6": "be9c1dd56c90db8ad3df64ad69fa691da31a313c1f9774c1c70fb91f8901aa56",
}


def make_case(case_id: str):
    """Load an immutable evaluation array; never regenerate it at run time."""
    import numpy as np

    path = FIXTURES / f"{case_id}.npy"
    data = np.load(path, allow_pickle=False)
    if data.dtype != np.float64 or data.shape != (CASES[case_id].n, CASES[case_id].p):
        raise RuntimeError(f"invalid frozen fixture metadata: {path}")
    data = np.ascontiguousarray(data)
    actual = array_sha256(data)
    if actual != EXPECTED_INPUT_SHA256[case_id]:
        raise RuntimeError(f"frozen fixture drift: {case_id}: {actual}")
    return data.copy()


def verify_installed_identity() -> dict[str, str]:
    import hashlib
    import os

    import pystatistics
    from pystatistics.mvnmle._dveb.loader import ABI_VERSION, LIBRARY_PATH, LIBRARY_SHA256

    site = Path(os.environ["DVEB_PORTABLE_SITE"]).resolve()
    package_path = Path(pystatistics.__file__).resolve()
    if not package_path.is_relative_to(site):
        raise RuntimeError(f"package import escaped installed site: {package_path} not below {site}")
    actual = hashlib.sha256(LIBRARY_PATH.read_bytes()).hexdigest()
    if ABI_VERSION != 1 or LIBRARY_SHA256 != EXPECTED_ARTIFACT_SHA256 or actual != EXPECTED_ARTIFACT_SHA256:
        raise RuntimeError(
            f"portable artifact identity mismatch: ABI={ABI_VERSION}, "
            f"manifest={LIBRARY_SHA256}, actual={actual}"
        )
    return {
        "site": str(site),
        "package_path": str(package_path),
        "artifact_path": str(LIBRARY_PATH.resolve()),
        "artifact_sha256": actual,
        "abi_version": str(ABI_VERSION),
    }
