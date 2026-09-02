#!/usr/bin/env python3
"""Assemble the admitted artifacts and zero-observation native campaign freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / "docs/research/evidence/dveb_arma_phase0b_native"
SOURCES = (
    "common.py",
    "reference_impl.py",
    "torch_impl.py",
    "native_runtime.py",
    "build_native.py",
    "derive_native_tolerances.py",
    "qualify_native.py",
    "proposal_trace.py",
    "freeze_proposals.py",
    "calibrate_native.py",
    "campaign_existing.py",
    "run_l3_campaign.py",
    "native/arma_native.h",
    "native/arma_cpu.cpp",
    "native/arma_cuda.cu",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*arguments: str) -> str:
    return subprocess.check_output(arguments, text=True, stderr=subprocess.STDOUT).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admitted-dir", type=Path, required=True)
    parser.add_argument("--freeze-name", default="native-freeze.json")
    parser.add_argument("--invalidated-attempts", type=int, default=0)
    parser.add_argument("--invalidated-reason")
    args = parser.parse_args()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    copies = {
        "build.json": args.admitted_dir / "build.json",
        "tolerance-derivation.json": args.admitted_dir / "tolerance-derivation.json",
        "qualification.json": args.admitted_dir / "qualification.json",
        "calibration.json": args.admitted_dir / "calibration.json",
        "libarma_native_cpu.so": args.admitted_dir / "libarma_native_cpu.so",
        "libarma_native_cuda.so": args.admitted_dir / "libarma_native_cuda.so",
    }
    for name, source in copies.items():
        if not source.is_file():
            raise SystemExit(f"missing admitted input: {source}")
        shutil.copy2(source, EVIDENCE / name)

    calibration = json.loads((EVIDENCE / "calibration.json").read_text())
    qualification = json.loads((EVIDENCE / "qualification.json").read_text())
    tolerances = json.loads((EVIDENCE / "tolerance-derivation.json").read_text())
    if calibration["decision_observations"] != 0:
        raise SystemExit("calibration contains decision observations")
    if qualification["status"] != "pass" or tolerances["status"] != "pass":
        raise SystemExit("native correctness admission failed")

    cpu_library = EVIDENCE / "libarma_native_cpu.so"
    cuda_library = EVIDENCE / "libarma_native_cuda.so"
    result = {
        "schema": "pystatistics.dveb-arma-phase0b.native-freeze.v1",
        "status": "frozen-before-decision-timing",
        "branch": command("git", "-C", str(ROOT), "branch", "--show-current"),
        "source_commit": command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "source_tree": command("git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"),
        "protocol_sha256": sha256(ROOT / "docs/research/DVEB_ARMA_PHASE0B_PROTOCOL.md"),
        "design_sha256": sha256(ROOT / "docs/research/DVEB_ARMA_PHASE0B_NATIVE_DESIGN.md"),
        "input_freeze_sha256": sha256(HERE / "input-freeze.json"),
        "proposal_freeze_sha256": sha256(HERE / "proposal-freeze.json"),
        "sources": {relative: sha256(HERE / relative) for relative in SOURCES},
        "evidence": {
            name: sha256(EVIDENCE / name)
            for name in (
                "build.json",
                "tolerance-derivation.json",
                "qualification.json",
                "calibration.json",
            )
        },
        "artifacts": {
            "cpu": {
                "path": "libarma_native_cpu.so",
                "sha256": sha256(cpu_library),
                "bytes": cpu_library.stat().st_size,
                "dependencies": command("ldd", str(cpu_library)),
            },
            "cuda": {
                "path": "libarma_native_cuda.so",
                "sha256": sha256(cuda_library),
                "bytes": cuda_library.stat().st_size,
                "dependencies": command("ldd", str(cuda_library)),
                "resource_usage": command(
                    "/home/haishuo/cuda-13.0/bin/cuobjdump",
                    "--dump-resource-usage",
                    str(cuda_library),
                ),
            },
        },
        "environment": {
            "platform": platform.platform(),
            "gxx": command("g++", "--version").splitlines()[0],
            "nvcc": command("/home/haishuo/cuda-13.0/bin/nvcc", "--version").splitlines()[-1],
            "gpu": command(
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader",
            ),
        },
        "selected_cuda_blocks": calibration["block_mapping_by_r"],
        "derived_floor_relative": tolerances["derived_floor_relative"],
        "decision_observations": 0,
        "invalidated_attempts": args.invalidated_attempts,
        "invalidated_reason": args.invalidated_reason,
        "native_diagnostic_authorized": True,
        "dveb_compiler_work_authorized": False,
    }
    (EVIDENCE / args.freeze_name).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"evidence": str(EVIDENCE), "artifacts": result["artifacts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
