#!/usr/bin/env python3
"""Build the separately authored exact-ARMA native diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "native"
DEFAULT_BUILD = ROOT / "build/dveb_arma_phase0b_native"
NVCC = Path("/home/haishuo/cuda-13.0/bin/nvcc")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> tuple[str, float]:
    started = time.perf_counter()
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return completed.stdout + completed.stderr, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    cpu = args.build_dir / "libarma_native_cpu.so"
    cuda = args.build_dir / "libarma_native_cuda.so"
    cpu_command = [
        "g++",
        "-std=c++17",
        "-O3",
        "-march=native",
        "-fopenmp",
        "-ffp-contract=off",
        "-fPIC",
        "-shared",
        str(SOURCE / "arma_cpu.cpp"),
        "-o",
        str(cpu),
    ]
    cuda_command = [
        str(NVCC),
        "-std=c++17",
        "-O3",
        "-arch=sm_120",
        "--fmad=true",
        "-Xcompiler=-fPIC",
        "-shared",
        str(SOURCE / "arma_cuda.cu"),
        "-o",
        str(cuda),
    ]
    cpu_output, cpu_seconds = run(cpu_command)
    cuda_output, cuda_seconds = run(cuda_command)
    result = {
        "schema": "pystatistics.dveb-arma-phase0b.native-build.v1",
        "platform": platform.platform(),
        "commands": {"cpu": cpu_command, "cuda": cuda_command},
        "seconds": {"cpu": cpu_seconds, "cuda": cuda_seconds},
        "compiler_output": {"cpu": cpu_output, "cuda": cuda_output},
        "sources": {
            path.name: sha256(path)
            for path in (SOURCE / "arma_native.h", SOURCE / "arma_cpu.cpp", SOURCE / "arma_cuda.cu")
        },
        "artifacts": {
            "cpu": {"path": str(cpu), "sha256": sha256(cpu), "bytes": cpu.stat().st_size},
            "cuda": {"path": str(cuda), "sha256": sha256(cuda), "bytes": cuda.stat().st_size},
        },
    }
    output = args.output or args.build_dir / "build.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "artifacts": result["artifacts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
