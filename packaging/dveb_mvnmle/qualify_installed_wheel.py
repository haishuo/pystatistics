#!/usr/bin/env python3
"""Standalone qualification of an installed CPU-only PyStatistics wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()

    if importlib.util.find_spec("torch") is not None or "torch" in sys.modules:
        raise SystemExit("qualification environment contains PyTorch")

    from pystatistics.mvnmle import datasets, mlest
    from pystatistics.mvnmle._dveb.loader import (
        ABI_VERSION,
        LIBRARY_PATH,
        LIBRARY_SHA256,
        MANIFEST_PATH,
        SCHEDULE_AUTO,
        SCHEDULE_SERIAL,
        SCHEDULE_WORK_ITEM_PARALLEL,
        _linux_cpu_flags,
        _missing_x86_64_v2_features,
    )
    from pystatistics.mvnmle._dveb.objective import DVEBDenseObjective

    metadata = importlib.metadata.metadata("pystatistics")
    requirements = metadata.get_all("Requires-Dist") or []
    forbidden_requirements = [
        item
        for item in requirements
        if any(word in item.lower() for word in ("torch", "cuda", "nvidia"))
    ]
    if forbidden_requirements:
        raise SystemExit(f"forbidden required dependencies: {forbidden_requirements}")

    manifest = json.loads(MANIFEST_PATH.read_text())
    if ABI_VERSION != 1 or manifest["abi_version"] != 1:
        raise SystemExit("ABI version mismatch")
    if (
        sha256(LIBRARY_PATH) != LIBRARY_SHA256
        or manifest["artifact_sha256"] != LIBRARY_SHA256
    ):
        raise SystemExit("artifact identity mismatch")

    flags = _linux_cpu_flags()
    missing = sorted(_missing_x86_64_v2_features(flags))
    if missing:
        raise SystemExit(f"qualification CPU does not satisfy x86-64-v2: {missing}")

    ldd = subprocess.run(
        ["ldd", str(LIBRARY_PATH)], check=True, capture_output=True, text=True
    ).stdout
    lowered = ldd.lower()
    if "not found" in lowered or any(word in lowered for word in ("torch", "cuda", "nvidia")):
        raise SystemExit(f"forbidden or unresolved ELF dependency:\n{ldd}")
    dynamic = subprocess.run(
        ["readelf", "-d", str(LIBRARY_PATH)], check=True, capture_output=True, text=True
    ).stdout
    for line in dynamic.splitlines():
        has_search_path = "RPATH" in line or "RUNPATH" in line
        if has_search_path and (
            "$ORIGIN" not in line or "/mnt/" in line or "/home/" in line
        ):
            raise SystemExit(f"unsafe artifact search path: {line}")

    schedule_outputs = []
    for schedule in (SCHEDULE_SERIAL, SCHEDULE_WORK_ITEM_PARALLEL, SCHEDULE_AUTO):
        objective = DVEBDenseObjective(datasets.apple, threads=6, schedule=schedule)
        theta = objective.get_initial_parameters()
        value, gradient = objective.compute_value_and_gradient(theta)
        schedule_outputs.append(
            {
                "schedule": schedule,
                "selected": objective.last_selected_schedule,
                "value": value,
                "gradient": gradient.copy(),
            }
        )
        objective.close()
    reference_output = schedule_outputs[0]
    for output in schedule_outputs[1:]:
        if abs(output["value"] - reference_output["value"]) > 1.0e-12:
            raise SystemExit("schedule objective mismatch")
        np.testing.assert_allclose(
            output["gradient"], reference_output["gradient"], atol=1.0e-12, rtol=0.0
        )

    data = datasets.apple.copy()
    before = data.tobytes()
    result = mlest(data, method="direct", backend="cpu", solver="dveb")
    if data.tobytes() != before:
        raise SystemExit("caller-owned input changed")
    reference = json.loads(args.reference.read_text())
    if not result.converged or result.backend_name != "cpu_dveb_cholesky_fp64":
        raise SystemExit("public DVEB fit did not converge with the expected backend")
    if abs(result.loglik - reference["loglik"]) >= 1.0e-7:
        raise SystemExit("public DVEB log-likelihood missed the frozen reference")
    np.testing.assert_allclose(result.muhat, reference["muhat"], rtol=1.0e-3)
    np.testing.assert_allclose(result.sigmahat, reference["sigmahat"], rtol=1.0e-3)
    if result.info["dveb_abi_version"] != 1:
        raise SystemExit("public result ABI identity mismatch")
    if result.info["dveb_artifact_sha256"] != LIBRARY_SHA256:
        raise SystemExit("public result artifact identity mismatch")

    second = mlest(
        datasets.missvals.copy(), method="direct", backend="cpu", solver="dveb"
    )
    if not math.isfinite(second.loglik):
        raise SystemExit("second dataset returned a non-finite result")

    if importlib.util.find_spec("torch") is not None or "torch" in sys.modules:
        raise SystemExit("PyTorch appeared during the DVEB fit")

    report = {
        "status": "pass",
        "python": sys.version,
        "wheel_version": metadata["Version"],
        "requirements": requirements,
        "torch_present": False,
        "abi_version": ABI_VERSION,
        "artifact": str(LIBRARY_PATH),
        "artifact_sha256": LIBRARY_SHA256,
        "artifact_size_bytes": LIBRARY_PATH.stat().st_size,
        "cpu_flags_found": sorted(flags or ()),
        "x86_64_v2_missing": missing,
        "ldd": ldd.splitlines(),
        "dynamic": dynamic.splitlines(),
        "schedules": [
            {
                "requested": row["schedule"],
                "selected": row["selected"],
                "value": row["value"],
                "bitwise_value_equal_to_serial": (
                    np.float64(row["value"]).tobytes()
                    == np.float64(reference_output["value"]).tobytes()
                ),
                "bitwise_gradient_equal_to_serial": bool(
                    np.array_equal(
                        row["gradient"].view(np.uint64),
                        reference_output["gradient"].view(np.uint64),
                    )
                ),
            }
            for row in schedule_outputs
        ],
        "apple": {
            "converged": result.converged,
            "loglik": result.loglik,
            "input_unchanged": True,
        },
        "missvals": {"converged": second.converged, "loglik": second.loglik},
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
