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
make_case = _MODULE.make_case
solution_record = _MODULE.solution_record

SEED = 0xD0EB1009
EXPECTED_ARTIFACT_SHA256 = (
    "65438a0d742257acc0dc7bd2ff13d2017669fd21dc07f64add41b18fb229fe38"
)


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
